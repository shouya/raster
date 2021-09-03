use std::mem;

use nalgebra::Point2;

pub fn point2_to_pixel(
  p: &Point2<f32>,
  size: (usize, usize),
) -> Option<(usize, usize)> {
  let x = p.x.round() as i32;
  let y = p.y.round() as i32;
  if x < 0 || x >= size.0 as i32 || y < 0 || y >= size.1 as i32 {
    return None;
  }

  Some((x as usize, y as usize))
}

pub fn sorted_tuple2<T>(mut a: (T, T)) -> (T, T)
where
  T: Ord + Copy,
{
  if a.0 > a.1 {
    mem::swap(&mut a.0, &mut a.1);
  }
  a
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
