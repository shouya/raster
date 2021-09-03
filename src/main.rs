use std::f32::consts::PI;

use eframe::{self, egui, epi};
use nalgebra::Point3;
use raster::{Camera, Mesh, Rasterizer, Scene};

mod raster;
mod util;

use crate::raster::Image;

struct RasterApp {
  texture_size: (usize, usize),
  texture_handle: Option<egui::TextureId>,
}

impl Default for RasterApp {
  fn default() -> Self {
    Self {
      texture_size: (600, 400),
      texture_handle: None,
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

impl epi::App for RasterApp {
  fn name(&self) -> &str {
    "Toy rasterizer"
  }

  fn update(&mut self, ctx: &egui::CtxRef, frame: &mut epi::Frame) {
    egui::containers::panel::CentralPanel::default().show(ctx, |ui| {
      ctx.settings_ui(ui);
      ui.label("Hello world");

      if let Some(texture_id) = self.texture_handle {
        let texture_size =
          (self.texture_size.0 as f32, self.texture_size.1 as f32);
        ui.image(texture_id, texture_size);
      } else {
        let image = {
          let scene = sample_scene();
          let mut raster = Rasterizer::new(self.texture_size);
          raster.rasterize(&scene);
          raster.into_image()
        };

        let texture_data = convert_texture(&image);
        let texture_id = frame.tex_allocator().alloc_srgba_premultiplied(
          self.texture_size,
          texture_data.as_slice(),
        );
        self.texture_handle = Some(texture_id);
        let texture_size_f =
          (self.texture_size.0 as f32, self.texture_size.1 as f32);
        ui.image(texture_id, texture_size_f);
      }
    });
  }
}

fn main() {
  eframe::run_native(Box::new(RasterApp::default()), Default::default());
}

fn sample_scene() -> Scene {
  let fov = 130.0 / 360.0 * 2.0 * PI;
  let camera = Camera::new_perspective(1.0, fov, -1.0, -50.0);
  let mut scene = Scene::new(camera);
  scene.add_mesh(Mesh::new_quad([
    Point3::new(-1.0, -1.0, -2.0),
    Point3::new(-1.0, 1.0, -3.0),
    Point3::new(1.0, 1.0, -3.0),
    Point3::new(1.0, -1.0, -3.0),
  ]));

  scene
}
