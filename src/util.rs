use std::cmp;

use crate::types::{Vec3, Vec4};

// copied from https://doc.rust-lang.org/src/core/num/f32.rs.html#959
pub fn f32_cmp(a: &f32, b: &f32) -> cmp::Ordering {
  let mut left = a.to_bits() as i32;
  let mut right = b.to_bits() as i32;
  left ^= (((left >> 31) as u32) >> 1) as i32;
  right ^= (((right >> 31) as u32) >> 1) as i32;

  left.cmp(&right)
}

pub fn reflect(l: &Vec3, n: &Vec3) -> Vec3 {
  *n * -2.0 * n.dot(*l) + *l
}

pub fn divw(v: Vec4) -> Vec4 {
  v / Vec4::splat(v.w)
}

pub fn divw3(v: Vec4) -> Vec3 {
  v.truncate() / Vec3::splat(v.w)
}
