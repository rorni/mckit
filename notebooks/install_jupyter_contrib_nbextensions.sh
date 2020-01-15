#!/bin/bash

# See Will Koehresen, Jupyter Notebook Extensions 
# https://towardsdatascience.com/jupyter-notebook-extensions-517fa69d2231
# Original: https://gist.github.com/luiscape/24546988ba27dbb7c92b7d3e710b604b


install_dependency() {
    conda install --yes $1 || pip install $1
}


install_dependencies() {

    local libs_to_install="\
        autopep8           \
        line_profiler      \
        memory_profiler    \
        pandas             \
        pandas-data-reader \
        scikit-learn       \
        seaborn            \
        netcdf4            \
        pillow             \
    "

    local installed=0
    local specified=0

    for lib in $libs_to_install; do
        let specified++
        install_dependency $lib && let installed++
    done

    return $(( $specified == $installed ))
}


install_nbextensions() {
    echo "Installing Jupyter extensions"
    conda install -c conda-forge jupyter_contrib_nbextensions
    jupyter contrib nbextension install --user
    jupyter nbextensions_configurator enable --user
}


install_theme_library() {
    pip install jupyterthemes
}

set_jupyter_theme() {
    local default_theme=${1:-grade3}
    jt -t $default_theme
}

main() {
    install_dependencies  && \
    install_nbextensions  && \
    install_theme_library && \
    set_jupyter_theme
}


[ -z "$BATS_PREFIX" ] &&  main


