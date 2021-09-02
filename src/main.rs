use eframe::{self, egui, epi};

mod raster;

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
        let image = Image::new(self.texture_size);
        let texture_data = convert_texture(&image);
        let texture_id = frame.tex_allocator().alloc_srgba_premultiplied(
          self.texture_size,
          texture_data.as_slice(),
        );
        let texture_size =
          (self.texture_size.0 as f32, self.texture_size.1 as f32);
        // rasterize(scene, &mut image)
        self.texture_handle = Some(texture_id);
        ui.image(texture_id, texture_size);
      }
    });
  }
}

fn main() {
  eframe::run_native(Box::new(RasterApp::default()), Default::default());
}
