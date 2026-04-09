using Documenter
using H2Matrices

makedocs(;
    sitename = "H2Matrices.jl",
    modules  = [H2Matrices],
    authors  = "Yiming Zhang",
    warnonly = [:missing_docs],
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://duserzym.github.io/H2Matrices.jl",
        assets     = String[],
    ),
    pages = [
        "Home"            => "index.md",
        "Theory"          => "theory.md",
        "Getting Started" => "getting_started.md",
        "Examples"        => "examples.md",
        "API Reference"   => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/duserzym/H2Matrices.jl.git",
    devbranch = "main",
)
