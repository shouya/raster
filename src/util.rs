use std::mem;

use nalgebra::Point2;

pub fn point2_to_pixel(p: &Point2<f32>) -> (i32, i32) {
  let x = p.x.round() as i32;
  let y = p.y.round() as i32;
  (x, y)
}

pub fn sorted_tuple3<T>(mut a: (T, T, T)) -> (T, T, T)
where
  T: Ord + Copy,
{
  if a.0 > a.1 {
    mem::swap(&mut a.0, &mut a.1);
  }
  if a.1 > a.2 {
    mem::swap(&mut a.1, &mut a.2);
  }
  if a.0 > a.1 {
    mem::swap(&mut a.0, &mut a.1);
  }
  a
}

pub fn lerp(r: f32, a: f32, b: f32) -> f32 {
  (b - a) * r + a
}

pub fn lerp_int(r: f32, a: i32, b: i32) -> i32 {
  a + ((b - a) as f32 * r).round() as i32
}
