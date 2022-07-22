using VlasovPoissonTwoSpecies
using Documenter

DocMeta.setdocmeta!(VlasovPoissonTwoSpecies, :DocTestSetup, :(using VlasovPoissonTwoSpecies); recursive=true)

makedocs(;
    modules=[VlasovPoissonTwoSpecies],
    authors="Julia Vlasov",
    repo="https://github.com/pnavaro/VlasovPoissonTwoSpecies.jl/blob/{commit}{path}#{line}",
    sitename="VlasovPoissonTwoSpecies.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pnavaro.github.io/VlasovPoissonTwoSpecies.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pnavaro/VlasovPoissonTwoSpecies.jl",
    devbranch="main",
)
