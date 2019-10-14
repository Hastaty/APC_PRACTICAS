#!/bin/bash
# ./make-delivery aaa/bbb <- Not aaa/bbb/
if [[ -z "${1}" ]]
then 
	echo 'Parameters needed'
	echo "./make-delivery 'project name' 'parent project dir' 'path to some docs'"
	exit
fi	
#### Copy project
project_name="${1}"
parent_project_dir="${2}" # The directory where you see this project
scripts_dir="$(pwd)"
cd "${parent_project_dir}"
mkdir tmp
cd tmp
mkdir "${project_name}"
cd "${project_name}"
cp -r  "${parent_project_dir}/${project_name}/" .
cd "${scripts_dir}"
mv "${parent_project_dir}/tmp/${project_name}/" .
rm -d "${parent_project_dir}/tmp/"
#### Filter and prepare project
cd "${project_name}/${project_name}"
rm -r .git/
if (("$project_name" == 'regression_classifier'))
then
	cd docs
	rm -r dataset_web_files/
	rm dataset_web.html
	cd ../..
	cp -r "${3}" .
	cd ..
fi
#### Zip
zip -r "${project_name}.zip" "${project_name}" 
rm -r "${project_name}"
