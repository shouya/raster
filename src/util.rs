use std::cmp;

use nalgebra::Vector3;

// copied from https://doc.rust-lang.org/src/core/num/f32.rs.html#959
pub fn f32_cmp(a: &f32, b: &f32) -> cmp::Ordering {
  let mut left = a.to_bits() as i32;
  let mut right = b.to_bits() as i32;
  left ^= (((left >> 31) as u32) >> 1) as i32;
  right ^= (((right >> 31) as u32) >> 1) as i32;

  left.cmp(&right)
}

pub fn reflect(l: &Vector3<f32>, n: &Vector3<f32>) -> Vector3<f32> {
  -2.0 * n.dot(&l) * n + l
}

pub fn avg3(
  a: &Vector3<f32>,
  b: &Vector3<f32>,
  c: &Vector3<f32>,
) -> Vector3<f32> {
  a + b + c / 3.0
}
