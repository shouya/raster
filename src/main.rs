use std::{
  collections::HashMap,
  f32::consts::PI,
  fs::read_dir,
  path::{Path, PathBuf},
};

use eframe::{
  self,
  egui::{
    self, color_picker::show_color, CollapsingHeader, ComboBox, Grid, Vec2,
  },
  epi::{self, TextureAllocator},
  NativeOptions,
};
use glam::{EulerRot, Quat};
use raster::{
  Camera, Color, Rasterizer, RasterizerMetric, RasterizerMode, Scene,
  WorldMesh, COLOR,
};
use shader::{Light, ShaderOptions};
use wavefront::MeshObject;

mod lerp;
mod raster;
mod shader;
mod types;
mod util;
mod wavefront;

use crate::{
  raster::Image,
  shader::TextureFilterMode,
  types::{Mat4, Point3, Vector3},
};

pub struct RenderResult {
  pub image: Image<Color>,
  pub zbuf_image: Image<Color>,
  pub metric: RasterizerMetric,
}

struct SceneCache {
  stash: HashMap<PathBuf, MeshObject>,
}

impl SceneCache {
  fn new() -> Self {
    Self {
      stash: HashMap::new(),
    }
  }

  fn get_mesh_obj<'a>(&'a mut self, path: &Path) -> &'a MeshObject {
    self
      .stash
      .entry(path.to_path_buf())
      .or_insert_with(move || Self::load_wavefront_meshes(path))
  }

  fn load_wavefront_meshes(path: &Path) -> MeshObject {
    wavefront::load(path).unwrap()
  }
}

struct RasterApp {
  texture_size: (usize, usize),
  texture_handle: Option<egui::TextureId>,
  render_result: Option<RenderResult>,
  redraw: bool,
  models: Vec<PathBuf>,
  tunable: Tunable,
  cache: SceneCache,
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
      cache: SceneCache::new(),
    }
  }
}

pub struct Tunable {
  distance: f32,
  fov: f32,
  znear: f32,
  zfar: f32,
  rot: [f32; 3],
  trans: [f32; 3],
  mode: RasterizerMode,
  zbuffer_mode: bool,
  model_file: PathBuf,
  double_faced: bool,
  shader_options: ShaderOptions,
  super_sampling: f32,
}

impl Default for Tunable {
  fn default() -> Self {
    Self {
      distance: 5.0,
      fov: 100.0,
      znear: 0.1,
      zfar: 1000.0,
      rot: [0.783, -0.3635, -0.1202],
      trans: [0.0, 0.0, 4.4579],
      shader_options: Default::default(),
      mode: RasterizerMode::Shaded,
      zbuffer_mode: false,
      model_file: "assets/chair.obj".into(),
      double_faced: false,
      super_sampling: 1.0,
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
    CollapsingHeader::new("Camera control")
      .default_open(true)
      .show(ui, |ui| {
        self.draw_camera_control(ui);
      });

    CollapsingHeader::new("Model selection")
      .default_open(true)
      .show(ui, |ui| {
        self.draw_model_selection(ui);
      });

    CollapsingHeader::new("Model transformation")
      .default_open(true)
      .show(ui, |ui| {
        self.draw_model_rotation(ui);
        self.draw_model_translation(ui);
      });

    CollapsingHeader::new("Shader options")
      .default_open(true)
      .show(ui, |ui| {
        self.draw_shader_options(ui);
      });
  }

  fn draw_camera_control(&mut self, ui: &mut egui::Ui) {
    let t = &mut self.tunable;
    let sliders = [
      (&mut t.distance, -10.0, 100.0, "Distance"),
      (&mut t.fov, 10.0, 180.0, "FoV"),
      (&mut t.znear, -100.0, 100.0, "Z near"),
      (&mut t.zfar, -100.0, 100.0, "Z far"),
    ];

    for (v, mi, ma, t) in sliders {
      if ui.add(egui::Slider::new(v, mi..=ma).text(t)).changed() {
        self.redraw = true;
      }
    }
  }

  fn draw_model_rotation(&mut self, ui: &mut egui::Ui) {
    ui.horizontal_top(|ui| {
      let rot = &mut self.tunable.rot;
      for i in 0..=2 {
        if ui
          .add(egui::DragValue::new(&mut rot[i]).speed(0.01))
          .changed()
        {
          self.redraw = true;
        }
      }
      ui.label("Rotation");
    });
  }

  fn draw_model_translation(&mut self, ui: &mut egui::Ui) {
    ui.horizontal_top(|ui| {
      let trans = &mut self.tunable.trans;
      for i in 0..=2 {
        if ui
          .add(egui::DragValue::new(&mut trans[i]).speed(0.01))
          .changed()
        {
          self.redraw = true;
        }
      }
      ui.label("Translation");
    });
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

    if ui
      .checkbox(&mut t.double_faced, "Double-faced mesh")
      .changed()
    {
      self.redraw = true;
    }

    use TextureFilterMode::*;
    let texture_filter_modes =
      [(Nearest, "Nearest pixel"), (Bilinear, "Bilinear")];

    ui.horizontal(|ui| {
      let t = &mut self.tunable.shader_options.texture_filter_mode;
      for (mode, text) in texture_filter_modes {
        if ui.radio_value(t, mode, text).clicked() {
          self.redraw = true;
        }
      }
    });

    ui.horizontal(|ui| {
      let t = &mut self.tunable.super_sampling;
      ui.label("Supersampling");
      for n in [0.5, 1.0, 2.0, 3.0] {
        if ui.radio_value(t, n, format!("{:.1}x", n)).clicked() {
          self.redraw = true;
        }
      }
    });
  }

  fn draw_model_selection(&mut self, ui: &mut egui::Ui) {
    let t = &mut self.tunable;
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
      self.handle_interative_transform(&resp);

      let topleft = resp.rect.min;
      if let Some(mut pos) = resp.hover_pos() {
        pos -= topleft.to_vec2();
        resp.on_hover_ui_at_pointer(|ui| {
          self.draw_tooltip_ui(ui, pos);
        });
      }
    }
  }

  fn handle_interative_transform(&mut self, resp: &egui::Response) {
    use egui::PointerButton::*;

    let size = (self.texture_size.0 as f32, self.texture_size.1 as f32);
    let d = resp.drag_delta();

    const ROTATION_SPEED: f32 = 1.5;
    const TRANSLATION_SPEED: f32 = 0.01;

    if resp.dragged() {
      // rotation
      if resp.dragged_by(Primary) {
        let t = &self.tunable;
        let rot_delta = Mat4::from_euler(
          EulerRot::XYZ,
          d.y / size.1 * ROTATION_SPEED,
          d.x / size.0 * ROTATION_SPEED,
          0.0,
        );
        let rot_old =
          Mat4::from_euler(EulerRot::XYZ, t.rot[0], t.rot[1], t.rot[2]);
        let rot_new =
          Quat::from_mat4(&(rot_delta * rot_old)).to_euler(EulerRot::XYZ);
        self.tunable.rot[0] = rot_new.0;
        self.tunable.rot[1] = rot_new.1;
        self.tunable.rot[2] = rot_new.2;
        self.redraw = true;
      }

      // translation
      if resp.dragged_by(Middle) {
        self.tunable.trans[0] += d.x * TRANSLATION_SPEED;
        self.tunable.trans[1] -= d.y * TRANSLATION_SPEED;
        self.redraw = true;
      }
    }

    // zooming
    let scroll_y = resp.ctx.input().scroll_delta.y;
    if scroll_y != 0.0 {
      self.tunable.trans[2] += scroll_y * TRANSLATION_SPEED;
      self.redraw = true;
    }
  }

  fn redraw_texture(&mut self, tex_alloc: &mut dyn TextureAllocator) {
    if !self.redraw && self.texture_handle.is_some() {
      return;
    }

    let result = {
      let scene = sample_scene(&self.tunable, &mut self.cache);
      render_scene(&self.tunable, self.texture_size, &scene)
    };

    if let Some(texture_id) = self.texture_handle {
      tex_alloc.free(texture_id);
    }

    let image_size = result.image.size();
    let texture_data = if self.tunable.zbuffer_mode {
      convert_texture(&result.zbuf_image)
    } else {
      convert_texture(&result.image)
    };

    let texture_id =
      tex_alloc.alloc_srgba_premultiplied(image_size, texture_data.as_slice());

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

  pub fn draw_tooltip_ui(&self, ui: &mut egui::Ui, pos: egui::Pos2) {
    let scale = self.tunable.super_sampling;
    let coords = ((pos.x * scale) as i32, (pos.y * scale) as i32);
    let render_result = self.render_result.as_ref().unwrap();
    let image_pixel = render_result.image.pixel(coords);
    let zbuf_pixel = render_result.zbuf_image.pixel(coords);

    ui.label(format!("{},{}", coords.0, coords.1));

    if let Some(color) = image_pixel {
      ui.horizontal(|ui| {
        let egui_color = egui::color::Rgba::from_rgb(color.x, color.y, color.z);
        show_color(ui, egui_color, Vec2::new(8.0, 8.0));
        ui.label(format!("{:.2},{:.2},{:.2}", color.x, color.y, color.z));
      });
    }

    if let Some(color) = zbuf_pixel {
      ui.label(format!("depth: {:.10}", color.x));
    }
  }

  pub fn draw_metrics(&self, ui: &mut egui::Ui) {
    if self.render_result.is_none() {
      return;
    }
    let metric = &self.render_result.as_ref().unwrap().metric;

    Grid::new("metrics")
      .num_columns(2)
      .striped(true)
      .show(ui, |ui| {
        ui.label("Render time");
        ui.label(format!(
          "{:.4}s ({:.2}fps)",
          metric.render_time,
          1.0 / metric.render_time
        ));
        ui.end_row();

        ui.label("Faces rendered");
        ui.label(format!("{}", metric.faces_rendered));
        ui.end_row();

        let msg = format!(
          "{}\n{}",
          "Clipped: number of triangles after clipping",
          "Sub-trig: number of upper+lower triangles"
        );
        ui.label("Triangles rendered (clipped, sub-trig)")
          .on_hover_text(msg);
        ui.label(format!(
          "{}, ({}, {})",
          metric.triangles_rendered,
          metric.clipped_triangles_rendered,
          metric.sub_triangles_rendered
        ));
        ui.end_row();

        ui.label("Hidden faces removed");
        ui.label(format!("{}", metric.hidden_face_removed));
        ui.end_row();

        ui.label("Lines drawn (horizontal)");
        ui.label(format!(
          "{} ({})",
          metric.lines_drawn, metric.horizontal_lines_drawn
        ));
        ui.end_row();

        ui.label("Vertices/pixels shaded");
        ui.label(format!(
          "{} / {}",
          metric.vertices_shaded, metric.pixels_shaded
        ));
        ui.end_row();

        ui.label("Pixels discarded")
          .on_hover_text("Due to lower z-value");

        ui.label(format!("{}", metric.pixels_discarded));
        ui.end_row();
      });
  }
}

impl epi::App for RasterApp {
  fn name(&self) -> &str {
    "Toy rasterizer"
  }

  fn update(&mut self, ctx: &egui::CtxRef, frame: &mut epi::Frame) {
    egui::SidePanel::left("misc").show(ctx, |ui| {
      egui::TopBottomPanel::bottom("metrics").show_inside(ui, |ui| {
        self.draw_metrics(ui);
      });
      egui::CentralPanel::default().show_inside(ui, |ui| {
        self.draw_tunables(ui);
      });
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
  let real_size = (
    (size.0 as f32 * tun.super_sampling) as usize,
    (size.1 as f32 * tun.super_sampling) as usize,
  );
  let mut raster = Rasterizer::new(real_size);
  raster.set_mode(tun.mode);
  raster.set_shader_options(tun.shader_options.clone());
  raster.rasterize(&scene);

  RenderResult {
    metric: raster.metric(),
    zbuf_image: raster.zbuffer_image(),
    image: raster.into_image(),
  }
}

fn sample_scene<'a>(tun: &'a Tunable, cache: &'a mut SceneCache) -> Scene<'a> {
  let fov = tun.fov / 360.0 * PI;
  let zfar = if tun.znear == tun.zfar {
    // avoid znear == zfar
    tun.znear + 1.0
  } else {
    tun.zfar
  };

  let mut camera = Camera::new_perspective(16.0 / 9.0, fov, tun.znear, zfar);
  let cam_trans =
    Mat4::from_translation(Vector3::new(0.0, 0.0, tun.distance).into());
  camera.transformd(&cam_trans);

  // let rotation = camera
  //   .matrix()
  //   .pseudo_inverse(0.001)
  //   .unwrap()
  //   .transform_vector(&Vector3::new(tun.rot_horizontal, tun.rot_vertical, 0.0));
  let mut scene = Scene::new(camera);

  let translation = Mat4::from_translation(tun.trans.into());
  let rotation =
    Mat4::from_euler(EulerRot::XYZ, tun.rot[0], tun.rot[1], tun.rot[2]);

  let mesh_obj = cache.get_mesh_obj(&tun.model_file);
  scene.set_texture_stash(&mesh_obj.textures);
  for mesh in mesh_obj.meshes.iter() {
    let mesh = WorldMesh::from(mesh)
      .transformed(rotation)
      .transformed(translation)
      .double_faced(tun.double_faced);
    scene.add_mesh(mesh);
  }

  scene.add_light(Light::new(
    Point3::new(5.0, 10.0, 5.0),
    COLOR::rgb(1.0, 1.0, 1.0),
  ));

  scene
}

fn bench_render() {
  let tun = Tunable::default();
  let mut cache = SceneCache::new();
  const N: usize = 20;

  for _ in 0..N {
    let scene = sample_scene(&tun, &mut cache);
    render_scene(&tun, (600, 400), &scene);
  }
}
