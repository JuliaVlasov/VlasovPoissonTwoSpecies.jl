using VlasovPoissonTwoSpecies
using Documenter

DocMeta.setdocmeta!(VlasovPoissonTwoSpecies, :DocTestSetup, :(using VlasovPoissonTwoSpecies); recursive=true)

makedocs(;
    modules=[VlasovPoissonTwoSpecies],
    authors="Julia Vlasov",
    repo="https://github.com/juliavlasov/VlasovPoissonTwoSpecies.jl/blob/{commit}{path}#{line}",
    sitename="VlasovPoissonTwoSpecies.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliavlasov.github.io/VlasovPoissonTwoSpecies.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaVlasov/VlasovPoissonTwoSpecies.jl",
    devbranch="main",
)
