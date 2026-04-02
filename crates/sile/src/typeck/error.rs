#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeError {
    message: String,
}

impl TypeError {
    pub fn unsupported_expr(kind: &str) -> Self {
        Self {
            message: format!("unsupported expression kind: {kind}"),
        }
    }

    pub fn unsupported_builtin(name: impl Into<String>) -> Self {
        Self {
            message: format!("unsupported builtin: {}", name.into()),
        }
    }
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for TypeError {}
