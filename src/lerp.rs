use crate::types::{Vector2, Vector3, Vector4};

pub trait Lerp {
  fn lerp(&self, other: &Self, t: f32) -> Self;
}

impl Lerp for f32 {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    self + (other - self) * t
  }
}

impl Lerp for i32 {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    self + ((other - self) as f32 * t) as Self
  }
}

pub struct Uniform<T> {
  pub value: T,
}

impl<T> Lerp for Uniform<T>
where
  T: Clone,
{
  fn lerp(&self, other: &Self, t: f32) -> Self {
    if t <= 0.5 {
      Self {
        value: self.value.clone(),
      }
    } else {
      Self {
        value: other.value.clone(),
      }
    }
  }
}

impl<T: Lerp> Lerp for Option<T> {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    if let Some(a) = self.as_ref() {
      if let Some(b) = other.as_ref() {
        return Some(lerp(t, a, b));
      }
    }
    None
  }
}

#[inline]
pub fn lerp<T>(t: f32, from: &T, to: &T) -> T
where
  T: Lerp,
{
  from.lerp(to, t)
}

struct LerpIter<T> {
  from: T,
  to: T,
  curr: usize,
  max: usize,
}

impl<T> Iterator for LerpIter<T>
where
  T: Lerp,
{
  type Item = T;
  fn next(&mut self) -> Option<T> {
    if self.curr > self.max {
      return None;
    }
    let t = if self.max != 0 {
      self.curr as f32 / self.max as f32
    } else {
      0.5
    };
    let v = lerp(t, &self.from, &self.to);
    self.curr += 1;
    Some(v)
  }
}

// includes both ends
pub fn lerp_closed_iter<T>(
  from: T,
  to: T,
  count: usize,
) -> impl Iterator<Item = T>
where
  T: Lerp,
{
  debug_assert!(count > 0);
  LerpIter {
    from,
    to,
    curr: 0,
    max: count,
  }
}

impl Lerp for Vector4 {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    Self::lerp(*self, *other, t)
  }
}

impl Lerp for Vector3 {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    Self::lerp(*self, *other, t)
  }
}

// impl Lerp for Point3 {}
//
// Point3 is the same type as Vector3 under different alias so we
// don't need to implement it

impl Lerp for Vector2 {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    Self::lerp(*self, *other, t)
  }
}
