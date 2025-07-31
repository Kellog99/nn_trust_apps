# NN Trust Applications

This repository collects all works related or based on nn_trust core attack_library


## Submodules management

Create .gitmodules file at the top level of the repo

### Add a submodule

Example adding nn_trust as a submodule, using the `main` branch
`git submodule add -b main git@github.com:LeoPhilosophers/nn_trust.git`

### Update a submodule branch

An exaple setting develop for the branch of attack-server/submodules/nn_trust submodule
`git config -f .gitmodules submodule.attack-server/submodules/nn_trust.branch develop
`