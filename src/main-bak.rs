use std::{f32::consts::PI, path::Path};

use async_std::task::{self, JoinHandle};
use eframe::{
  self,
  egui::{self, ProgressBar},
  epi::{self, TextureAllocator},
};
use futures::{TryFuture, future::lazy};
use nalgebra::{Matrix4, Vector3};
use raster::{Camera, Mesh, Rasterizer, RasterizerMode, Scene};
use wavefront::Wavefront;

mod lerp;
mod raster;
mod shader;
mod util;
mod wavefront;

use crate::raster::Image;

struct RasterApp {
  texture_size: (usize, usize),
  texture_handle: Option<egui::TextureId>,
  image: Option<Image>,
  pending: Option<JoinHandle<Image>>,
  tunable: Tunable,
}

impl Default for RasterApp {
  fn default() -> Self {
    Self {
      texture_size: (600, 400),
      texture_handle: None,
      image: None,
      pending: None,
      tunable: Tunable::default(),
    }
  }
}

#[derive(Clone, Debug)]
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
}

impl Default for Tunable {
  fn default() -> Self {
    Self {
      distance: 5.0,
      fov: 100.0,
      znear: 1.0,
      zfar: 1000.0,
      rot_x: 0.1,
      rot_y: 0.3,
      rot_z: 0.0,
      trans_x: 0.0,
      trans_y: 0.0,
      trans_z: 4.3,
      mode: RasterizerMode::Wireframe,
    }
  }
}

fn convert_texture(image: &Image) -> Vec<egui::Color32> {
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
    let mut redraw = false;
    let t = &mut self.tunable;
    let sliders = [
      (&mut t.distance, -10.0, 100.0, "Distance"),
      (&mut t.fov, 10.0, 180.0, "FoV"),
      (&mut t.znear, -100.0, 100.0, "Z near"),
      (&mut t.zfar, -100.0, 100.0, "Z far"),
      (&mut t.rot_x, -2.0 * PI, 2.0 * PI, "Rotation (X)"),
      (&mut t.rot_y, -2.0 * PI, 2.0 * PI, "Rotation (Y)"),
      (&mut t.rot_z, -2.0 * PI, 2.0 * PI, "Rotation (Z)"),
      (&mut t.trans_x, -5.0, 5.0, "Translation (X)"),
      (&mut t.trans_y, -5.0, 5.0, "Translation (Y)"),
      (&mut t.trans_z, -5.0, 5.0, "Translation (Z)"),
    ];

    for (v, mi, ma, t) in sliders {
      if ui.add(egui::Slider::new(v, mi..=ma).text(t)).changed() {
        redraw = true;
      }
    }

    ui.horizontal(|ui| {
      use RasterizerMode::*;

      let t = &mut self.tunable;
      if (ui.radio_value(&mut t.mode, Wireframe, "Wireframe")).clicked() {
        redraw = true;
      }
      if (ui.radio_value(&mut t.mode, Shaded, "Shaded")).clicked() {
        redraw = true;
      }
      if (ui.radio_value(&mut t.mode, Zbuffer, "Zbuffer")).clicked() {
        redraw = true;
      }
    });

    if redraw {
      self.redraw();
    }
  }

  fn draw_canvas(&self, ui: &mut egui::Ui) {
    let size = (self.texture_size.0 as f32, self.texture_size.1 as f32);
    if let Some(texture_id) = self.texture_handle {
      let resp = ui.image(texture_id, size);
      let topleft = resp.rect.min;
      if let Some(mut pos) = resp.hover_pos() {
        pos -= topleft.to_vec2();
        resp.on_hover_ui_at_pointer(|ui| {
          let coords = (pos.x as i32, pos.y as i32);
          ui.label(format!("{},{}", coords.0, coords.1));
          if let Some(color) = self.image.as_ref().unwrap().pixel(coords) {
            ui.label(format!("{:.2},{:.2},{:.2}", color.x, color.y, color.z));
          }
        });
      }
    }

    if let Some(handle) = self.pending {
      ui.add(
        ProgressBar::new(0.5)
          .animate(true)
          .text("Generation in progress"),
      );
    }
  }

  fn redraw(&mut self) {
    if let Some(handle) = self.pending {
      handle.cancel().ok()
    }

    let tunable = self.tunable.clone();
    self.pending = task::spawn(lazy(|_| {
      let scene = sample_scene(&tunable);
      render_scene(&tunable, self.texture_size, &scene)
    }));
  }

  fn load_texture(
    &mut self,
    image: &Image,
    tex_alloc: &mut dyn TextureAllocator,
  ) {
    let texture_data = convert_texture(&image);
    if let Some(texture_id) = self.texture_handle {
      tex_alloc.free(texture_id);
    }
    let texture_id = tex_alloc
      .alloc_srgba_premultiplied(self.texture_size, texture_data.as_slice());
    self.texture_handle = Some(texture_id);
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
      if let Some(handle) = self.pending {
        if let Some(image) = TryFuture::po handle.try_poll() {
          self.load_texture(&image, frame.tex_allocator());
          self.image = Some(image);
          self.pending = None;
        }
      }

      self.draw_canvas(ui);
    });
  }
}

fn main() {
  eframe::run_native(Box::new(RasterApp::default()), Default::default());
}

fn render_scene(tun: &Tunable, size: (usize, usize), scene: &Scene) -> Image {
  let mut raster = Rasterizer::new(size);
  raster.set_mode(tun.mode);
  raster.rasterize(&scene);
  raster.into_image()
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

  let mut scene = Scene::new(camera);
  let rotation = Vector3::new(tun.rot_x, tun.rot_y, tun.rot_z);
  let translation = Vector3::new(tun.trans_x, tun.trans_y, tun.trans_z);

  let wavefront =
    Wavefront::from_file(Path::new("assets/pumpkin.obj")).unwrap();

  scene.add_mesh(
    Mesh::new_wavefront(wavefront)
      .transformed(Matrix4::new_rotation(rotation))
      .transformed(Matrix4::new_translation(&translation)),
  );

  // scene.add_mesh(Mesh::new_quad([
  //   Point3::new(-1.0, -1.0, -1.0),
  //   Point3::new(-1.0, 1.0, -1.0),
  //   Point3::new(1.0, 1.0, -1.0),
  //   Point3::new(1.0, -1.0, -1.0),
  // ]));

  scene
}