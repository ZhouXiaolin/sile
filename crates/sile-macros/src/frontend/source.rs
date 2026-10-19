//! Where an entry was written.
//!
//! The Tile IR alignment work needs the kernel's source: cutile-rs prints its
//! Tile IR in the order its front end walks the arguments of each call, so a
//! lowering that wants the same text has to replay that walk. The module's
//! token text is enough for that — the walk depends on which expressions the
//! source names, not on how they were laid out — and `TokenStream::to_string`
//! is stable, so no file is read and no unstable API is used.

use proc_macro2::TokenStream;
use quote::ToTokens;
use quote::quote;

/// One entry's place in its module.
pub struct Origin {
    pub module: String,
    pub source: String,
    pub entry_index: usize,
}

impl Origin {
    /// The `KernelOrigin` literal for this place.
    pub fn to_tokens(&self) -> TokenStream {
        let module = &self.module;
        let source = &self.source;
        let entry_index = self.entry_index;
        quote! {
            ::sile::hir::KernelOrigin {
                module: #module.to_string(),
                source: #source.to_string(),
                entry_index: #entry_index,
            }
        }
    }

    /// The source text of a module body, as one canonical string.
    pub fn source_of(input: &syn::ItemMod) -> String {
        input.to_token_stream().to_string()
    }
}
