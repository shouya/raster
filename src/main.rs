use std::f32::consts::PI;

use eframe::{self, egui, epi};
use nalgebra::{Matrix4, Point3, Vector2, Vector3};
use raster::{Camera, Mesh, Rasterizer, Scene};

mod raster;
mod util;

use crate::raster::Image;

struct RasterApp {
  texture_size: (usize, usize),
  texture_handle: Option<egui::TextureId>,
  redraw: bool,
  tunable: Tunable,
}

impl Default for RasterApp {
  fn default() -> Self {
    Self {
      texture_size: (600, 400),
      texture_handle: None,
      redraw: true,
      tunable: Tunable::default(),
    }
  }
}

pub struct Tunable {
  distance: f32,
  fov: f32,
  rot_x: f32,
  rot_y: f32,
  rot_z: f32,
  trans_x: f32,
  trans_y: f32,
  trans_z: f32
}

impl Default for Tunable {
  fn default() -> Self {
    Self {
      distance: 10.0,
      fov: 100.0,
      rot_x: 0.0,
      rot_y: 0.0,
      rot_z: 0.0,
      trans_x: 0.0,
      trans_y: 0.0,
      trans_z: 0.0
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
    egui::SidePanel::left("tunable").show(ctx, |ui| {
      let sliders = [
        (&mut self.tunable.distance, -10.0, 100.0, "Distance"),
        (&mut self.tunable.fov, 10.0, 180.0, "FoV"),
        (&mut self.tunable.rot_x, -2.0*PI, 2.0*PI, "Rotation (X)"),
        (&mut self.tunable.rot_y, -2.0*PI, 2.0*PI, "Rotation (Y)"),
        (&mut self.tunable.rot_z, -2.0*PI, 2.0*PI, "Rotation (Z)"),
      ];

      for (v, mi, ma, t) in sliders {
        if ui.add(egui::Slider::new(v, mi..=ma).text(t)).changed() {
          self.redraw = true;
        }
      }
    });

    egui::CentralPanel::default().show(ctx, |ui| {
      if self.texture_handle.is_none() || self.redraw {
        let image = {
          let scene = sample_scene(&self.tunable);
          let mut raster = Rasterizer::new(self.texture_size);
          raster.rasterize(&scene);
          raster.into_image()
        };

        let texture_data = convert_texture(&image);
        if let Some(texture_id) = self.texture_handle {
          frame.tex_allocator().free(texture_id);
        }
        let texture_id = frame.tex_allocator().alloc_srgba_premultiplied(
          self.texture_size,
          texture_data.as_slice(),
        );
        self.texture_handle = Some(texture_id);
        self.redraw = false;
      }

      if let Some(texture_id) = self.texture_handle {
        let texture_size =
          (self.texture_size.0 as f32, self.texture_size.1 as f32);
        ui.image(texture_id, texture_size);
      }
    });
  }
}

fn main() {
  eframe::run_native(Box::new(RasterApp::default()), Default::default());
}

fn sample_scene(tunable: &Tunable) -> Scene {
  let fov = tunable.fov / 360.0 * 2.0 * PI;
  let mut camera = Camera::new_perspective(16.0 / 9.0, fov, -50.0, -1.0);
  let cam_rot = Matrix4::new_nonuniform_scaling(&Vector3::new(-1.0, 1.0, 1.0));
  let cam_trans = Matrix4::new_translation(&Vector3::new(
    0.0,
    0.0,
    tunable.distance,
  ));
  camera.transform(&(cam_rot * cam_trans));

  let mut scene = Scene::new(camera);
  let rotation = Vector3::new(tunable.rot_x, tunable.rot_y, tunable.rot_z);
  scene.add_mesh(Mesh::new_cube().transformed(Matrix4::new_rotation(rotation)));

  // scene.add_mesh(Mesh::new_quad([
  //   Point3::new(-1.0, -1.0, -1.0),
  //   Point3::new(-1.0, 1.0, -1.0),
  //   Point3::new(1.0, 1.0, -1.0),
  //   Point3::new(1.0, -1.0, -1.0),
  // ]));

  scene
}
