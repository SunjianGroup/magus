_magus_complete()
{
    local file_list=('summary' 'relax')
    if [[ $COMP_CWORD == 1 ]]; then 
        COMPREPLY=($(compgen -W 'summary search relax prepare clean' -- ${COMP_WORDS[COMP_CWORD]}))
    elif [[ ${file_list[@]} =~ ${COMP_WORDS[1]} ]]; then
        compopt -o filenames
        COMPREPLY=($(compgen -f -- ${COMP_WORDS[COMP_CWORD]}))
    fi
}
complete -F _magus_complete magus
