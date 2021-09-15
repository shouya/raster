use std::{f32::consts::PI, fs::read_dir, path::PathBuf};

use eframe::{
  self,
  egui::{self, ComboBox, Vec2},
  epi::{self, TextureAllocator},
  NativeOptions,
};
use nalgebra::{Matrix4, Point3, Rotation, Vector3};
use raster::{
  Camera, Color, Mesh, Rasterizer, RasterizerMetric, RasterizerMode, Scene,
  COLOR,
};
use shader::SpecularShader;
use wavefront::Wavefront;

mod lerp;
mod raster;
mod shader;
mod util;
mod wavefront;

use crate::raster::Image;

pub struct RenderResult {
  pub image: Image<Color>,
  pub zbuf_image: Image<Color>,
  pub metric: RasterizerMetric,
}

struct RasterApp {
  texture_size: (usize, usize),
  texture_handle: Option<egui::TextureId>,
  render_result: Option<RenderResult>,
  redraw: bool,
  models: Vec<PathBuf>,
  tunable: Tunable,
}

impl Default for RasterApp {
  fn default() -> Self {
    Self {
      texture_size: (600, 400),
      texture_handle: None,
      render_result: None,
      redraw: true,
      tunable: Tunable::default(),
      models: vec![],
    }
  }
}

pub struct Tunable {
  distance: f32,
  fov: f32,
  znear: f32,
  zfar: f32,
  rot_x: f32,
  rot_y: f32,
  rot_z: f32,
  trans_x: f32,
  trans_y: f32,
  trans_z: f32,
  mode: RasterizerMode,
  zbuffer_mode: bool,
  model_file: PathBuf,
  double_faced: bool,
}

impl Default for Tunable {
  fn default() -> Self {
    Self {
      distance: 5.0,
      fov: 100.0,
      znear: 1.0,
      zfar: 1000.0,
      rot_x: 0.0,
      rot_y: 0.0,
      rot_z: 0.0,
      trans_x: 0.0,
      trans_y: 0.0,
      trans_z: 3.0,
      mode: RasterizerMode::Shaded,
      zbuffer_mode: false,
      model_file: "assets/pumpkin_nosubdiv.obj".into(),
      double_faced: false,
    }
  }
}

fn convert_texture(image: &Image<Color>) -> Vec<egui::Color32> {
  let dim = image.width() * image.height();
  let mut pixel_cache = Vec::with_capacity(dim);
  let to256 = |f: f32| (f.clamp(0.0, 1.0) * 255.0).round() as u8;
  for pixel in image.pixels() {
    let (r, g, b) = (to256(pixel[0]), to256(pixel[1]), to256(pixel[2]));
    let color = egui::Color32::from_rgb(r, g, b);
    pixel_cache.push(color);
  }
  pixel_cache
}

impl RasterApp {
  fn draw_tunables(&mut self, ui: &mut egui::Ui) {
    let t = &mut self.tunable;
    let sliders = [
      (&mut t.distance, -10.0, 100.0, "Distance"),
      (&mut t.fov, 10.0, 180.0, "FoV"),
      (&mut t.znear, -100.0, 100.0, "Z near"),
      (&mut t.zfar, -100.0, 100.0, "Z far"),
      (&mut t.rot_x, -2.0 * PI, 2.0 * PI, "Rotation (X)"),
      (&mut t.rot_y, -2.0 * PI, 2.0 * PI, "Rotation (X)"),
      (&mut t.rot_z, -2.0 * PI, 2.0 * PI, "Rotation (Z)"),
      (&mut t.trans_x, -5.0, 5.0, "Translation (X)"),
      (&mut t.trans_y, -5.0, 5.0, "Translation (Y)"),
      (&mut t.trans_z, -5.0, 5.0, "Translation (Z)"),
    ];

    for (v, mi, ma, t) in sliders {
      if ui.add(egui::Slider::new(v, mi..=ma).text(t)).changed() {
        self.redraw = true;
      }
    }

    ui.separator();

    self.draw_shader_options(ui);
  }

  fn draw_shader_options(&mut self, ui: &mut egui::Ui) {
    use RasterizerMode::*;
    let modes = [
      (Wireframe, "Wireframe"),
      (Shaded, "Shaded"),
      (Clipped, "Clipped"),
    ];

    ui.horizontal(|ui| {
      let t = &mut self.tunable;
      for (mode, text) in modes {
        if ui.radio_value(&mut t.mode, mode, text).clicked() {
          self.redraw = true;
        }
      }
    });

    let t = &mut self.tunable;
    if ui.checkbox(&mut t.zbuffer_mode, "Z-buffer mode").changed() {
      self.redraw = true;
    }

    let model_file_name = t.model_file.file_stem().unwrap().to_str().unwrap();

    ComboBox::from_label("Select model")
      .selected_text(model_file_name)
      .show_ui(ui, |ui| {
        for file in self.models.iter() {
          let selected = &mut self.tunable.model_file;
          let value = file.to_path_buf();
          let text = file.file_stem().unwrap().to_str().unwrap();
          if ui.selectable_value(selected, value, text).changed() {
            self.redraw = true;
          }
        }
      });

    let t = &mut self.tunable;
    if ui
      .checkbox(&mut t.double_faced, "Double-faced mesh")
      .changed()
    {
      self.redraw = true;
    }
  }

  fn draw_canvas(&mut self, ui: &mut egui::Ui) {
    let size = (self.texture_size.0 as f32, self.texture_size.1 as f32);
    if let Some(texture_id) = self.texture_handle {
      let image = egui::Image::new(texture_id, size).sense(egui::Sense {
        click: true,
        drag: true,
        focusable: true,
      });
      let resp = ui.add(image);
      ui.label(format!("{:?}", self.render_result.as_ref().unwrap().metric));

      use egui::PointerButton::*;
      let d = resp.drag_delta();

      const ROTATION_SPEED: f32 = 1.5;
      const TRANSLATION_SPEED: f32 = 0.01;

      if resp.dragged() {
        if resp.dragged_by(Primary) {
          let t = &self.tunable;
          let rot_delta = Rotation::from_euler_angles(
            d.y / size.1 * ROTATION_SPEED,
            d.x / size.0 * ROTATION_SPEED,
            0.0,
          );
          let rot_old = Rotation::from_euler_angles(t.rot_x, t.rot_y, t.rot_z);
          let rot_new = (rot_delta * rot_old).euler_angles();
          self.tunable.rot_x = rot_new.0;
          self.tunable.rot_y = rot_new.1;
          self.tunable.rot_z = rot_new.2;
          self.redraw = true;
        }
        if resp.dragged_by(Middle) {
          self.tunable.trans_x += d.x * TRANSLATION_SPEED;
          self.tunable.trans_y -= d.y * TRANSLATION_SPEED;
          self.redraw = true;
        }
      }

      let scroll_y = resp.ctx.input().scroll_delta.y;
      if scroll_y != 0.0 {
        self.tunable.trans_z += scroll_y * TRANSLATION_SPEED;
        self.redraw = true;
      }

      let topleft = resp.rect.min;
      if let Some(mut pos) = resp.hover_pos() {
        pos -= topleft.to_vec2();
        resp.on_hover_ui_at_pointer(|ui| {
          let coords = (pos.x as i32, pos.y as i32);
          ui.label(format!("{},{}", coords.0, coords.1));
          if let Some(color) =
            self.render_result.as_ref().unwrap().image.pixel(coords)
          {
            ui.label(format!("{:.2},{:.2},{:.2}", color.x, color.y, color.z));
          }
          if let Some(color) = self
            .render_result
            .as_ref()
            .unwrap()
            .zbuf_image
            .pixel(coords)
          {
            ui.label(format!("depth: {:.2}", color.x));
          }
        });
      }
    }
  }

  fn redraw_texture(&mut self, tex_alloc: &mut dyn TextureAllocator) {
    if !self.redraw && self.texture_handle.is_some() {
      return;
    }

    let result = {
      let scene = sample_scene(&self.tunable);
      render_scene(&self.tunable, self.texture_size, &scene)
    };

    let texture_data = convert_texture(&result.image);
    if let Some(texture_id) = self.texture_handle {
      tex_alloc.free(texture_id);
    }
    let texture_id = tex_alloc
      .alloc_srgba_premultiplied(self.texture_size, texture_data.as_slice());

    self.texture_handle = Some(texture_id);
    self.render_result = Some(result);
    self.redraw = false;
  }

  fn load_models(&mut self) {
    let mut res = Vec::new();
    for entry in read_dir("assets").expect("./assets not found") {
      let path = entry.unwrap().path();
      if path.extension() == Some("obj".as_ref()) {
        res.push(path);
      }
    }

    self.models = res;
  }
}

impl epi::App for RasterApp {
  fn name(&self) -> &str {
    "Toy rasterizer"
  }

  fn update(&mut self, ctx: &egui::CtxRef, frame: &mut epi::Frame) {
    egui::SidePanel::left("tunable").show(ctx, |ui| {
      self.draw_tunables(ui);
    });

    egui::CentralPanel::default().show(ctx, |ui| {
      self.redraw_texture(frame.tex_allocator());
      self.draw_canvas(ui);
    });
  }

  fn setup(
    &mut self,
    _ctx: &egui::CtxRef,
    _frame: &mut epi::Frame<'_>,
    _storage: Option<&dyn epi::Storage>,
  ) {
    self.load_models()
  }
}

fn main() {
  let args: Vec<_> = std::env::args().collect();
  if args.len() > 1 && args[1] == "bench" {
    bench_render();
  } else {
    let mut options: NativeOptions = Default::default();
    options.initial_window_size = Some(Vec2::new(900.0, 600.0));
    eframe::run_native(Box::new(RasterApp::default()), options);
  }
}

fn render_scene(
  tun: &Tunable,
  size: (usize, usize),
  scene: &Scene,
) -> RenderResult {
  let mut raster = Rasterizer::new(size);
  raster.set_mode(tun.mode);
  raster.rasterize(&scene);

  RenderResult {
    metric: raster.metric(),
    zbuf_image: raster.zbuffer_image(),
    image: raster.into_image(),
  }
}

fn sample_scene(tun: &Tunable) -> Scene {
  let fov = tun.fov / 360.0 * PI;
  let zfar = if tun.znear == tun.zfar {
    // avoid znear == zfar
    tun.znear + 1.0
  } else {
    tun.zfar
  };

  let mut camera = Camera::new_perspective(16.0 / 9.0, fov, tun.znear, zfar);
  let cam_trans =
    Matrix4::new_translation(&Vector3::new(0.0, 0.0, tun.distance));
  camera.transformd(&cam_trans);

  // let rotation = camera
  //   .matrix()
  //   .pseudo_inverse(0.001)
  //   .unwrap()
  //   .transform_vector(&Vector3::new(tun.rot_horizontal, tun.rot_vertical, 0.0));
  let mut scene = Scene::new(camera);

  let translation = Vector3::new(tun.trans_x, tun.trans_y, tun.trans_z);
  let rotation = Rotation::from_euler_angles(tun.rot_x, tun.rot_y, tun.rot_z);

  let wavefront = Wavefront::from_file(&tun.model_file).unwrap();
  let shader =
    SpecularShader::new(COLOR::rgb(1.0, 0.5, 0.0), Point3::new(5.0, 10.0, 5.0));
  let mesh = Mesh::new_wavefront(wavefront)
    .transformed(rotation.to_homogeneous())
    .transformed(Matrix4::new_translation(&translation))
    .shaded(shader)
    .double_faced(tun.double_faced);

  scene.add_mesh(mesh);

  scene
}

fn bench_render() {
  let tun = Tunable::default();
  let scene = sample_scene(&tun);
  render_scene(&tun, (600, 400), &scene);
}
