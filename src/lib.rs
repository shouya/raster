mod lerp;
mod raster;
mod shader;
mod util;
mod wavefront;
mod app;

#[cfg(target_arch = "wasm32")]
use crate::app::RasterApp;

// ----------------------------------------------------------------------------
// When compiling for web:

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::{self, prelude::*};

#[cfg(target_arch = "wasm32")]
use console_error_panic_hook;

/// This is the entry-point for all the web-assembly.
/// This is called once from the HTML.
/// It loads the app, installs some callbacks, then returns.
/// You can add more callbacks like this if you want to call in to your code.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start(canvas_id: &str) -> Result<(), eframe::wasm_bindgen::JsValue> {
  std::panic::set_hook(Box::new(console_error_panic_hook::hook));

  let app = RasterApp::default();
  eframe::start_web(canvas_id, Box::new(app))
}
