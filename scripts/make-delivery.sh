#!/bin/bash
# .../code_project
# .../Documentos
project_dir='/mnt/c/Users/ricar_xuzvwqa/Codigo/regression_classifier/'
docs='/mnt/c/Users/ricar_xuzvwqa/OneDrive/Documentos/Asignaturas/Primer Semestre - 2019-2020/Aprendizaje Computacional/Prácticas/Documentos/'
project_dir="${project_dir%/}"
docs="${docs%/}"
project_name="${project_dir##*/}"
pushd . > /dev/null
dir="$(pwd)"
cd "${project_dir}"
dir="${dir#${project_dir}}"
dir="${dir#/}"
dir="./${dir}/${project_name}/"
git checkout-index -a --prefix "${dir}"
popd > /dev/null
cd "${project_name}" 
if (("$project_name" == 'regression_classifier'))
then
	rm -r docs/dataset_web_files/
	rm docs/dataset_web.html
fi
cd ..
zip -r "${project_name}.zip" "${project_name}" 
rm -r "${project_name}" 
if (("$project_name" == 'regression_classifier'))
then 
	dir="$(pwd)"
	pushd . > /dev/null
	cd "${docs%/*}"	
	zip -r "${dir}/${project_name}.zip" "${docs##*/}"
	popd > /dev/null
fi
