use std::{fs::File, io::Read, path::Path, str::FromStr};

use anyhow;
use nalgebra::{Point3, Vector2, Vector3};

use crate::raster::{Face, IndexedPolyVert};

pub struct Wavefront {
  pub vertices: Vec<Point3<f32>>,
  pub vertex_normals: Vec<Vector3<f32>>,
  pub texture_coords: Vec<Vector2<f32>>,
  pub faces: Vec<Face<IndexedPolyVert>>,
}

impl Wavefront {
  pub fn from_string(s: &str) -> anyhow::Result<Wavefront> {
    let mut vertices = Vec::new();
    let mut vertex_normals = Vec::new();
    let mut texture_coords = Vec::new();
    let mut faces = Vec::new();
    for line in s.lines() {
      if line.starts_with("#") {
        continue;
      }
      if line.trim().is_empty() {
        continue;
      }
      let segments = line.split(" ").collect::<Vec<_>>();
      match segments.as_slice() {
        ["v", vs @ ..] => {
          vertices.push(Point3::from_slice(&Self::parse_floats(vs)?))
        }
        ["vn", vns @ ..] => {
          let vec = Vector3::from_column_slice(&Self::parse_floats(vns)?);
          vertex_normals.push(vec)
        }
        ["vt", vts @ ..] => {
          let vec = Vector2::from_column_slice(&Self::parse_floats(vts)?);
          texture_coords.push(vec)
        }
        ["f", fs @ ..] => faces.push(Self::parse_face(fs)?),
        _ => {}
      }
    }

    let wf = Wavefront {
      vertices,
      vertex_normals,
      texture_coords,
      faces,
    };

    Ok(wf)
  }

  fn parse_floats(slices: &[&str]) -> anyhow::Result<Vec<f32>> {
    slices
      .iter()
      .map(|s| f32::from_str(s))
      .collect::<Result<Vec<_>, _>>()
      .map_err(anyhow::Error::from)
  }

  fn parse_face(slices: &[&str]) -> anyhow::Result<Face<IndexedPolyVert>> {
    let mut face = Face::new(false);
    for v in slices {
      let indices: Vec<Result<usize, _>> =
        v.split("/").map(|i| usize::from_str(i)).collect();

      match indices.as_slice() {
        [Ok(vi)] => face.add_vert(IndexedPolyVert::new(vi - 1)),
        [Ok(vi), Ok(ti)] => {
          face.add_vert(IndexedPolyVert::new_texture(vi - 1, ti - 1))
        }
        [Ok(vi), Ok(ti), Ok(ni)] => {
          face.add_vert(IndexedPolyVert::new_texture_normal(vi - 1, ti - 1, ni - 1))
        }
        [Ok(vi), Err(_), Ok(ni)] => {
          face.add_vert(IndexedPolyVert::new_normal(vi - 1, ni - 1))
        }
        _ => anyhow::bail!("Invalid face: {:?}", indices),
      }
    }

    Ok(face)
  }

  pub fn from_file(path: &Path) -> anyhow::Result<Wavefront> {
    let mut buf = String::new();
    File::open(path)?.read_to_string(&mut buf)?;
    Self::from_string(&buf)
  }
}
