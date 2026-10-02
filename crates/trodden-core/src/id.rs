use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

macro_rules! string_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
        )]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(id: impl Into<String>) -> Self {
                Self(id.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(id: String) -> Self {
                Self(id)
            }
        }

        impl From<$name> for String {
            fn from(id: $name) -> Self {
                id.0
            }
        }
    };
}

string_id! {
    SessionId
}

string_id! {
    HarnessId
}

string_id! {
    ProcedureId
}

string_id! {
    FamilyId
}

string_id! {
    RepoId
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_plain_string() {
        let id = SessionId::new("0b6f7c1e-2d4a-4f0e-9a51-3c8e2f1d7b90");

        let json = serde_json::to_string(&id).expect("string id serializes");

        assert_eq!(json, r#""0b6f7c1e-2d4a-4f0e-9a51-3c8e2f1d7b90""#);
    }
}
