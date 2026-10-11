//! Vertex attribute conversion from glTF accessors to engine format.
//!
//! This handles ALL glTF accessor formats (u8/u16/u32/f32, normalized or not,
//! scalar/vec2/vec3/vec4) and converts them to the engine's vertex format.
//! This fixes the "mesh explosion" bugs caused by incorrect index/vertex handling.

use std::collections::HashMap;

use gltf::{
    accessor::{DataType, Dimensions},
    mesh::util::{ReadColors, ReadJoints, ReadTexCoords, ReadWeights},
};
use thiserror::Error;

use crate::gltf_utils::convert_coordinates::ConvertCoordinates;

/// Represents whether integer data requires normalization
#[derive(Copy, Clone)]
struct Normalization(bool);

impl Normalization {
    fn apply_either<T, U>(
        self,
        value: T,
        normalized_ctor: impl Fn(T) -> U,
        unnormalized_ctor: impl Fn(T) -> U,
    ) -> U {
        if self.0 {
            normalized_ctor(value)
        } else {
            unnormalized_ctor(value)
        }
    }
}

/// An error that occurs when accessing buffer data
#[derive(Error, Debug)]
pub enum AccessFailed {
    /// Accessing the data failed because of an issue like a mismatch in stride,
    /// or a buffer view slice failing.
    #[error("Malformed vertex attribute data")]
    MalformedData,
    /// The format supplied is unsupported for this operation.
    #[error("Unsupported vertex attribute format")]
    UnsupportedFormat,
}

/// Helper for reading buffer data
struct BufferAccessor<'a> {
    accessor: gltf::Accessor<'a>,
    buffer_data: &'a [Vec<u8>],
    normalization: Normalization,
}

impl<'a> BufferAccessor<'a> {
    /// Creates an iterator over the elements in this accessor
    fn iter<T: gltf::accessor::Item>(self) -> Result<gltf::accessor::Iter<'a, T>, AccessFailed> {
        gltf::accessor::Iter::new(self.accessor, |buffer: gltf::Buffer| {
            self.buffer_data.get(buffer.index()).map(Vec::as_slice)
        })
        .ok_or(AccessFailed::MalformedData)
    }

    /// Applies the element iterator to a constructor or fails if normalization is required
    fn with_no_norm<T: gltf::accessor::Item, U>(
        self,
        ctor: impl Fn(gltf::accessor::Iter<'a, T>) -> U,
    ) -> Result<U, AccessFailed> {
        if self.normalization.0 {
            return Err(AccessFailed::UnsupportedFormat);
        }
        self.iter().map(ctor)
    }

    /// Applies the element iterator and the normalization flag to a constructor
    fn with_norm<T: gltf::accessor::Item, U>(
        self,
        ctor: impl Fn(gltf::accessor::Iter<'a, T>, Normalization) -> U,
    ) -> Result<U, AccessFailed> {
        let normalized = self.normalization;
        self.iter().map(|v| ctor(v, normalized))
    }
}

/// Engine vertex attribute value types (matches our GltfVertex/GltfGpuVertex needs)
#[derive(Debug, Clone)]
pub enum VertexAttributeValues {
    Float32(Vec<f32>),
    Uint32(Vec<u32>),
    Float32x2(Vec<[f32; 2]>),
    Uint32x2(Vec<[u32; 2]>),
    Float32x3(Vec<[f32; 3]>),
    Uint32x3(Vec<[u32; 3]>),
    Float32x4(Vec<[f32; 4]>),
    Uint32x4(Vec<[u32; 4]>),
    Snorm16x2(Vec<[i16; 2]>),
    Unorm16x2(Vec<[u16; 2]>),
    Sint16x2(Vec<[i16; 2]>),
    Uint16x2(Vec<[u16; 2]>),
    Snorm16x4(Vec<[i16; 4]>),
    Unorm16x4(Vec<[u16; 4]>),
    Sint16x4(Vec<[i16; 4]>),
    Uint16x4(Vec<[u16; 4]>),
    Snorm8x2(Vec<[i8; 2]>),
    Unorm8x2(Vec<[u8; 2]>),
    Sint8x2(Vec<[i8; 2]>),
    Uint8x2(Vec<[u8; 2]>),
    Snorm8x4(Vec<[i8; 4]>),
    Unorm8x4(Vec<[u8; 4]>),
    Sint8x4(Vec<[i8; 4]>),
    Uint8x4(Vec<[u8; 4]>),
}

impl VertexAttributeValues {
    pub fn len(&self) -> usize {
        match self {
            VertexAttributeValues::Float32(v) => v.len(),
            VertexAttributeValues::Uint32(v) => v.len(),
            VertexAttributeValues::Float32x2(v) => v.len(),
            VertexAttributeValues::Uint32x2(v) => v.len(),
            VertexAttributeValues::Float32x3(v) => v.len(),
            VertexAttributeValues::Uint32x3(v) => v.len(),
            VertexAttributeValues::Float32x4(v) => v.len(),
            VertexAttributeValues::Uint32x4(v) => v.len(),
            VertexAttributeValues::Snorm16x2(v) => v.len(),
            VertexAttributeValues::Unorm16x2(v) => v.len(),
            VertexAttributeValues::Sint16x2(v) => v.len(),
            VertexAttributeValues::Uint16x2(v) => v.len(),
            VertexAttributeValues::Snorm16x4(v) => v.len(),
            VertexAttributeValues::Unorm16x4(v) => v.len(),
            VertexAttributeValues::Sint16x4(v) => v.len(),
            VertexAttributeValues::Uint16x4(v) => v.len(),
            VertexAttributeValues::Snorm8x2(v) => v.len(),
            VertexAttributeValues::Unorm8x2(v) => v.len(),
            VertexAttributeValues::Sint8x2(v) => v.len(),
            VertexAttributeValues::Uint8x2(v) => v.len(),
            VertexAttributeValues::Snorm8x4(v) => v.len(),
            VertexAttributeValues::Unorm8x4(v) => v.len(),
            VertexAttributeValues::Sint8x4(v) => v.len(),
            VertexAttributeValues::Uint8x4(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An enum of the iterators used by different vertex attribute formats
enum VertexAttributeIter<'a> {
    // For reading native formats
    F32(gltf::accessor::Iter<'a, f32>),
    U32(gltf::accessor::Iter<'a, u32>),
    F32x2(gltf::accessor::Iter<'a, [f32; 2]>),
    U32x2(gltf::accessor::Iter<'a, [u32; 2]>),
    F32x3(gltf::accessor::Iter<'a, [f32; 3]>),
    U32x3(gltf::accessor::Iter<'a, [u32; 3]>),
    F32x4(gltf::accessor::Iter<'a, [f32; 4]>),
    U32x4(gltf::accessor::Iter<'a, [u32; 4]>),
    S16x2(gltf::accessor::Iter<'a, [i16; 2]>, Normalization),
    U16x2(gltf::accessor::Iter<'a, [u16; 2]>, Normalization),
    S16x4(gltf::accessor::Iter<'a, [i16; 4]>, Normalization),
    U16x4(gltf::accessor::Iter<'a, [u16; 4]>, Normalization),
    S8x2(gltf::accessor::Iter<'a, [i8; 2]>, Normalization),
    U8x2(gltf::accessor::Iter<'a, [u8; 2]>, Normalization),
    S8x4(gltf::accessor::Iter<'a, [i8; 4]>, Normalization),
    U8x4(gltf::accessor::Iter<'a, [u8; 4]>, Normalization),
    // Additional on-disk formats used for RGB colors
    U16x3(gltf::accessor::Iter<'a, [u16; 3]>, Normalization),
    U8x3(gltf::accessor::Iter<'a, [u8; 3]>, Normalization),
}

impl<'a> VertexAttributeIter<'a> {
    /// Creates an iterator over the elements in a vertex attribute accessor
    fn from_accessor(
        accessor: gltf::Accessor<'a>,
        buffer_data: &'a [Vec<u8>],
    ) -> Result<VertexAttributeIter<'a>, AccessFailed> {
        let normalization = Normalization(accessor.normalized());
        let format = (accessor.data_type(), accessor.dimensions());
        let acc = BufferAccessor {
            accessor,
            buffer_data,
            normalization,
        };
        match format {
            (DataType::F32, Dimensions::Scalar) => acc.with_no_norm(VertexAttributeIter::F32),
            (DataType::U32, Dimensions::Scalar) => acc.with_no_norm(VertexAttributeIter::U32),
            (DataType::F32, Dimensions::Vec2) => acc.with_no_norm(VertexAttributeIter::F32x2),
            (DataType::U32, Dimensions::Vec2) => acc.with_no_norm(VertexAttributeIter::U32x2),
            (DataType::F32, Dimensions::Vec3) => acc.with_no_norm(VertexAttributeIter::F32x3),
            (DataType::U32, Dimensions::Vec3) => acc.with_no_norm(VertexAttributeIter::U32x3),
            (DataType::F32, Dimensions::Vec4) => acc.with_no_norm(VertexAttributeIter::F32x4),
            (DataType::U32, Dimensions::Vec4) => acc.with_no_norm(VertexAttributeIter::U32x4),
            (DataType::I16, Dimensions::Vec2) => acc.with_norm(VertexAttributeIter::S16x2),
            (DataType::U16, Dimensions::Vec2) => acc.with_norm(VertexAttributeIter::U16x2),
            (DataType::I16, Dimensions::Vec4) => acc.with_norm(VertexAttributeIter::S16x4),
            (DataType::U16, Dimensions::Vec4) => acc.with_norm(VertexAttributeIter::U16x4),
            (DataType::I8, Dimensions::Vec2) => acc.with_norm(VertexAttributeIter::S8x2),
            (DataType::U8, Dimensions::Vec2) => acc.with_norm(VertexAttributeIter::U8x2),
            (DataType::I8, Dimensions::Vec4) => acc.with_norm(VertexAttributeIter::S8x4),
            (DataType::U8, Dimensions::Vec4) => acc.with_norm(VertexAttributeIter::U8x4),
            (DataType::U16, Dimensions::Vec3) => acc.with_norm(VertexAttributeIter::U16x3),
            (DataType::U8, Dimensions::Vec3) => acc.with_norm(VertexAttributeIter::U8x3),
            _ => Err(AccessFailed::UnsupportedFormat),
        }
    }

    /// Materializes values for any supported format of vertex attribute
    fn into_any_values(self, convert_coordinates: bool) -> Result<VertexAttributeValues, AccessFailed> {
        match self {
            VertexAttributeIter::F32(it) => Ok(VertexAttributeValues::Float32(it.collect())),
            VertexAttributeIter::U32(it) => Ok(VertexAttributeValues::Uint32(it.collect())),
            VertexAttributeIter::F32x2(it) => Ok(VertexAttributeValues::Float32x2(it.collect())),
            VertexAttributeIter::U32x2(it) => Ok(VertexAttributeValues::Uint32x2(it.collect())),
            VertexAttributeIter::F32x3(it) => Ok(if convert_coordinates {
                VertexAttributeValues::Float32x3(it.map(ConvertCoordinates::convert_coordinates).collect())
            } else {
                VertexAttributeValues::Float32x3(it.collect())
            }),
            VertexAttributeIter::U32x3(it) => Ok(VertexAttributeValues::Uint32x3(it.collect())),
            VertexAttributeIter::F32x4(it) => Ok(if convert_coordinates {
                VertexAttributeValues::Float32x4(it.map(ConvertCoordinates::convert_coordinates).collect())
            } else {
                VertexAttributeValues::Float32x4(it.collect())
            }),
            VertexAttributeIter::U32x4(it) => Ok(VertexAttributeValues::Uint32x4(it.collect())),
            VertexAttributeIter::S16x2(it, n) => {
                Ok(n.apply_either(it.collect(), VertexAttributeValues::Snorm16x2, VertexAttributeValues::Sint16x2))
            }
            VertexAttributeIter::U16x2(it, n) => {
                Ok(n.apply_either(it.collect(), VertexAttributeValues::Unorm16x2, VertexAttributeValues::Uint16x2))
            }
            VertexAttributeIter::S16x4(it, n) => {
                Ok(n.apply_either(it.collect(), VertexAttributeValues::Snorm16x4, VertexAttributeValues::Sint16x4))
            }
            VertexAttributeIter::U16x4(it, n) => {
                Ok(n.apply_either(it.collect(), VertexAttributeValues::Unorm16x4, VertexAttributeValues::Uint16x4))
            }
            VertexAttributeIter::S8x2(it, n) => {
                Ok(n.apply_either(it.collect(), VertexAttributeValues::Snorm8x2, VertexAttributeValues::Sint8x2))
            }
            VertexAttributeIter::U8x2(it, n) => {
                Ok(n.apply_either(it.collect(), VertexAttributeValues::Unorm8x2, VertexAttributeValues::Uint8x2))
            }
            VertexAttributeIter::S8x4(it, n) => {
                Ok(n.apply_either(it.collect(), VertexAttributeValues::Snorm8x4, VertexAttributeValues::Sint8x4))
            }
            VertexAttributeIter::U8x4(it, n) => {
                Ok(n.apply_either(it.collect(), VertexAttributeValues::Unorm8x4, VertexAttributeValues::Uint8x4))
            }
            _ => Err(AccessFailed::UnsupportedFormat),
        }
    }

    /// Materializes RGBA values, converting compatible formats to Float32x4
    fn into_rgba_values(self) -> Result<VertexAttributeValues, AccessFailed> {
        match self {
            VertexAttributeIter::U8x3(it, Normalization(true)) => Ok(VertexAttributeValues::Float32x4(
                ReadColors::RgbU8(it).into_rgba_f32().collect(),
            )),
            VertexAttributeIter::U16x3(it, Normalization(true)) => Ok(VertexAttributeValues::Float32x4(
                ReadColors::RgbU16(it).into_rgba_f32().collect(),
            )),
            VertexAttributeIter::F32x3(it) => Ok(VertexAttributeValues::Float32x4(
                ReadColors::RgbF32(it).into_rgba_f32().collect(),
            )),
            VertexAttributeIter::U8x4(it, Normalization(true)) => Ok(VertexAttributeValues::Float32x4(
                ReadColors::RgbaU8(it).into_rgba_f32().collect(),
            )),
            VertexAttributeIter::U16x4(it, Normalization(true)) => Ok(VertexAttributeValues::Float32x4(
                ReadColors::RgbaU16(it).into_rgba_f32().collect(),
            )),
            s => s.into_any_values(false),
        }
    }

    /// Materializes joint index values, converting compatible formats to Uint16x4
    fn into_joint_index_values(self) -> Result<VertexAttributeValues, AccessFailed> {
        match self {
            VertexAttributeIter::U8x4(it, Normalization(false)) => {
                Ok(VertexAttributeValues::Uint16x4(ReadJoints::U8(it).into_u16().collect()))
            }
            s => s.into_any_values(false),
        }
    }

    /// Materializes joint weight values, converting compatible formats to Float32x4
    fn into_joint_weight_values(self) -> Result<VertexAttributeValues, AccessFailed> {
        match self {
            VertexAttributeIter::U8x4(it, Normalization(true)) => {
                Ok(VertexAttributeValues::Float32x4(ReadWeights::U8(it).into_f32().collect()))
            }
            VertexAttributeIter::U16x4(it, Normalization(true)) => {
                Ok(VertexAttributeValues::Float32x4(ReadWeights::U16(it).into_f32().collect()))
            }
            s => s.into_any_values(false),
        }
    }

    /// Materializes texture coordinate values, converting compatible formats to Float32x2
    fn into_tex_coord_values(self) -> Result<VertexAttributeValues, AccessFailed> {
        match self {
            VertexAttributeIter::U8x2(it, Normalization(true)) => Ok(VertexAttributeValues::Float32x2(
                ReadTexCoords::U8(it).into_f32().collect(),
            )),
            VertexAttributeIter::U16x2(it, Normalization(true)) => Ok(VertexAttributeValues::Float32x2(
                ReadTexCoords::U16(it).into_f32().collect(),
            )),
            s => s.into_any_values(false),
        }
    }
}

enum ConversionMode {
    Any,
    Rgba,
    JointIndex,
    JointWeight,
    TexCoord,
}

/// Errors that can occur during the `convert_attribute` function.
#[derive(Error, Debug)]
pub enum ConvertAttributeError {
    /// The loaded format was different than the attribute's intended format.
    #[error("Vertex attribute {0} has format {1:?} but expected {3:?} for target attribute {2}")]
    WrongFormat(String, VertexFormat, String, VertexFormat),
    /// Fetching values from the glTF Accessor failed
    #[error("{0} in accessor {1}")]
    AccessFailed(AccessFailed, usize),
    /// A vertex attribute name was not one of the gltf crate's well-known attributes,
    /// nor was it registered by a user as a custom attribute. Therefore it is unknown.
    #[error("Unknown vertex attribute {0}")]
    UnknownName(String),
}

/// Vertex format for validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint8,
    Uint8x2,
    Uint8x4,
    Uint16,
    Uint16x2,
    Uint16x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint8,
    Sint8x2,
    Sint8x4,
    Sint16,
    Sint16x2,
    Sint16x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Unorm8,
    Unorm8x2,
    Unorm8x4,
    Unorm16,
    Unorm16x2,
    Unorm16x4,
    Snorm8,
    Snorm8x2,
    Snorm8x4,
    Snorm16,
    Snorm16x2,
    Snorm16x4,
}

impl From<&VertexAttributeValues> for VertexFormat {
    fn from(values: &VertexAttributeValues) -> Self {
        match values {
            VertexAttributeValues::Float32(_) => VertexFormat::Float32,
            VertexAttributeValues::Uint32(_) => VertexFormat::Uint32,
            VertexAttributeValues::Float32x2(_) => VertexFormat::Float32x2,
            VertexAttributeValues::Uint32x2(_) => VertexFormat::Uint32x2,
            VertexAttributeValues::Float32x3(_) => VertexFormat::Float32x3,
            VertexAttributeValues::Uint32x3(_) => VertexFormat::Uint32x3,
            VertexAttributeValues::Float32x4(_) => VertexFormat::Float32x4,
            VertexAttributeValues::Uint32x4(_) => VertexFormat::Uint32x4,
            VertexAttributeValues::Snorm16x2(_) => VertexFormat::Snorm16x2,
            VertexAttributeValues::Unorm16x2(_) => VertexFormat::Unorm16x2,
            VertexAttributeValues::Sint16x2(_) => VertexFormat::Sint16x2,
            VertexAttributeValues::Uint16x2(_) => VertexFormat::Uint16x2,
            VertexAttributeValues::Snorm16x4(_) => VertexFormat::Snorm16x4,
            VertexAttributeValues::Unorm16x4(_) => VertexFormat::Unorm16x4,
            VertexAttributeValues::Sint16x4(_) => VertexFormat::Sint16x4,
            VertexAttributeValues::Uint16x4(_) => VertexFormat::Uint16x4,
            VertexAttributeValues::Snorm8x2(_) => VertexFormat::Snorm8x2,
            VertexAttributeValues::Unorm8x2(_) => VertexFormat::Unorm8x2,
            VertexAttributeValues::Sint8x2(_) => VertexFormat::Sint8x2,
            VertexAttributeValues::Uint8x2(_) => VertexFormat::Uint8x2,
            VertexAttributeValues::Snorm8x4(_) => VertexFormat::Snorm8x4,
            VertexAttributeValues::Unorm8x4(_) => VertexFormat::Unorm8x4,
            VertexAttributeValues::Sint8x4(_) => VertexFormat::Sint8x4,
            VertexAttributeValues::Uint8x4(_) => VertexFormat::Uint8x4,
        }
    }
}

/// Engine vertex attribute definition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EngineVertexAttribute {
    pub name: &'static str,
    pub format: VertexFormat,
    pub offset: usize,
}

impl EngineVertexAttribute {
    pub const POSITION: Self = Self { name: "Position", format: VertexFormat::Float32x3, offset: 0 };
    pub const NORMAL: Self = Self { name: "Normal", format: VertexFormat::Float32x3, offset: 0 };
    pub const TANGENT: Self = Self { name: "Tangent", format: VertexFormat::Float32x4, offset: 0 };
    pub const UV_0: Self = Self { name: "UV_0", format: VertexFormat::Float32x2, offset: 0 };
    pub const UV_1: Self = Self { name: "UV_1", format: VertexFormat::Float32x2, offset: 0 };
    pub const COLOR: Self = Self { name: "Color", format: VertexFormat::Float32x4, offset: 0 };
    pub const JOINT_INDEX: Self = Self { name: "JointIndex", format: VertexFormat::Uint16x4, offset: 0 };
    pub const JOINT_WEIGHT: Self = Self { name: "JointWeight", format: VertexFormat::Float32x4, offset: 0 };
}

/// Map glTF vertex attributes into their engine format, optionally
/// converting values if necessary.
pub fn convert_attribute(
    semantic: gltf::Semantic,
    accessor: gltf::Accessor,
    buffer_data: &[Vec<u8>],
    custom_vertex_attributes: &HashMap<Box<str>, EngineVertexAttribute>,
    convert_coordinates: bool,
) -> Result<(EngineVertexAttribute, VertexAttributeValues), ConvertAttributeError> {
    if let Some((attribute, conversion, convert_coords)) = match &semantic {
        gltf::Semantic::Positions => Some((
            EngineVertexAttribute::POSITION,
            ConversionMode::Any,
            convert_coordinates,
        )),
        gltf::Semantic::Normals => Some((
            EngineVertexAttribute::NORMAL,
            ConversionMode::Any,
            convert_coordinates,
        )),
        gltf::Semantic::Tangents => Some((
            EngineVertexAttribute::TANGENT,
            ConversionMode::Any,
            convert_coordinates,
        )),
        gltf::Semantic::Colors(0) => Some((EngineVertexAttribute::COLOR, ConversionMode::Rgba, false)),
        gltf::Semantic::TexCoords(0) => {
            Some((EngineVertexAttribute::UV_0, ConversionMode::TexCoord, false))
        }
        gltf::Semantic::TexCoords(1) => {
            Some((EngineVertexAttribute::UV_1, ConversionMode::TexCoord, false))
        }
        gltf::Semantic::Joints(0) => Some((
            EngineVertexAttribute::JOINT_INDEX,
            ConversionMode::JointIndex,
            false,
        )),
        gltf::Semantic::Weights(0) => Some((
            EngineVertexAttribute::JOINT_WEIGHT,
            ConversionMode::JointWeight,
            false,
        )),
        _ => None,
    } {
        let raw_iter = VertexAttributeIter::from_accessor(accessor.clone(), buffer_data);
        let converted_values = raw_iter.and_then(|iter| match conversion {
            ConversionMode::Any => iter.into_any_values(convert_coords),
            ConversionMode::Rgba => iter.into_rgba_values(),
            ConversionMode::TexCoord => iter.into_tex_coord_values(),
            ConversionMode::JointIndex => iter.into_joint_index_values(),
            ConversionMode::JointWeight => iter.into_joint_weight_values(),
        });
        match converted_values {
            Ok(values) => {
                let loaded_format = VertexFormat::from(&values);
                if attribute.format == loaded_format {
                    Ok((attribute, values))
                } else {
                    Err(ConvertAttributeError::WrongFormat(
                        semantic.to_string(),
                        loaded_format,
                        attribute.name.to_string(),
                        attribute.format,
                    ))
                }
            }
            Err(err) => Err(ConvertAttributeError::AccessFailed(err, accessor.index())),
        }
    } else {
        Err(ConvertAttributeError::UnknownName(semantic.to_string()))
    }
}