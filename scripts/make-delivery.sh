#!/bin/bash
project_dir='/mnt/c/Users/ricar_xuzvwqa/Codigo/regression_classifier'
docs='/mnt/c/Users/ricar_xuzvwqa/OneDrive/Documentos/Asignaturas/Primer Semestre - 2019-2020/Aprendizaje Computacional/Prácticas/Documentos'
project_name="${project_dir##*/}"
project_dir2="${project_dir%/*}"
pushd . > /dev/null
cd "${project_dir}"
cd ..
mkdir tmp
cd tmp
cp -r "${project_dir}" .
popd > /dev/null 
mv "${project_dir2}/tmp/${project_name}" .
rm -d "${project_dir2}/tmp/"
cd "${project_name}"
rm -r .git/
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
