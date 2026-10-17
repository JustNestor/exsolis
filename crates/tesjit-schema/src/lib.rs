use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

#[proc_macro_derive(TsSchema)]
pub fn derive_ts_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive(&input) {
        Ok(output) => output.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

fn derive(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "TsSchema does not support generic types",
        ));
    }
    let name = &input.ident;
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    &data.fields,
                    "TsSchema requires a struct with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "TsSchema can only be derived for structs",
            ));
        }
    };
    let fields = fields
        .iter()
        .map(|field| {
            let ident = field.ident.as_ref().expect("named field");
            let ty = &field.ty;
            Ok(quote! {
                {
                    let (field_ty, field_spawnable) =
                        <#ty as crate::ts_ecs::TsSchemaField>::ts_field()?;
                    spawnable &= field_spawnable;
                    fields.push(::tesjit::StructField {
                        name: stringify!(#ident).to_owned(),
                        ty: field_ty,
                        offset: ::std::mem::offset_of!(Self, #ident) as u64,
                    });
                }
            })
        })
        .collect::<syn::Result<Vec<_>>>()?;

    Ok(quote! {
        impl crate::ts_ecs::TsSchema for #name {
            fn ts_schema() -> Result<::tesjit::StructLayout, String> {
                let mut fields = Vec::new();
                let mut spawnable = true;
                #(#fields)*
                let mut layout = ::tesjit::StructLayout::from_parts(
                    stringify!(#name),
                    fields,
                    ::std::mem::size_of::<Self>() as u64,
                    ::std::mem::align_of::<Self>() as u64,
                )?;
                layout.spawnable = spawnable;
                Ok(layout)
            }
        }

        impl crate::ts_ecs::TsSchemaField for #name {
            fn ts_field() -> Result<(::tesjit::Ty, bool), String> {
                let layout = <Self as crate::ts_ecs::TsSchema>::ts_schema()?;
                let spawnable = layout.spawnable;
                Ok((::tesjit::Ty::Struct(::std::sync::Arc::new(layout)), spawnable))
            }
        }
    })
}
