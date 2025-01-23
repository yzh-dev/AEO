#!/bin/bash

if [ "$#" -eq 1 ]; then
    outputPath="$1/EPIC_KITCHENS"
else
    outputPath="$HOME/Downloads/EPIC_KITCHENS"
fi

if ! [ -x "$(command -v wget)" ]; then
    echo "Error!: wget is not installed! Please install it and try again"
    exit 1
fi

echo -e "\n### ------------------------------------------------------- ###\n"
echo "### Downloading into $outputPath"
echo -e "\n### ------------------------------------------------------- ###\n"

epic_download() {
    local url=$1
    local path=$2
    local niceFilePath=$3
    
    echo -ne "# Downloading "$niceFilePath"\t"
    wget --progress=dot --continue --directory-prefix="$path" "$url" 2>&1 | grep --line-buffered "%" | sed -E "s,\.,,g" | awk '{printf("\b\b\b\b%4s", $2)}'
    echo -ne "\b\b\b\b"
    echo " # done"
}

epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P01/P01_11.tar" "$outputPath/frames_rgb_flow/flow/test/D2" "frames_rgb_flow/flow/test/D2/P01_11.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P01/P01_12.tar" "$outputPath/frames_rgb_flow/flow/test/D2" "frames_rgb_flow/flow/test/D2/P01_12.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P01/P01_13.tar" "$outputPath/frames_rgb_flow/flow/test/D2" "frames_rgb_flow/flow/test/D2/P01_13.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P01/P01_14.tar" "$outputPath/frames_rgb_flow/flow/test/D2" "frames_rgb_flow/flow/test/D2/P01_14.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P01/P01_15.tar" "$outputPath/frames_rgb_flow/flow/test/D2" "frames_rgb_flow/flow/test/D2/P01_15.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P08/P08_09.tar" "$outputPath/frames_rgb_flow/flow/test/D1" "frames_rgb_flow/flow/test/D1/P08_09.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P08/P08_10.tar" "$outputPath/frames_rgb_flow/flow/test/D1" "frames_rgb_flow/flow/test/D1/P08_10.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P08/P08_14.tar" "$outputPath/frames_rgb_flow/flow/test/D1" "frames_rgb_flow/flow/test/D1/P08_14.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P08/P08_15.tar" "$outputPath/frames_rgb_flow/flow/test/D1" "frames_rgb_flow/flow/test/D1/P08_15.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P08/P08_16.tar" "$outputPath/frames_rgb_flow/flow/test/D1" "frames_rgb_flow/flow/test/D1/P08_16.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P08/P08_17.tar" "$outputPath/frames_rgb_flow/flow/test/D1" "frames_rgb_flow/flow/test/D1/P08_17.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P22/P22_01.tar" "$outputPath/frames_rgb_flow/flow/test/D3" "frames_rgb_flow/flow/test/D3/P22_01.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P22/P22_02.tar" "$outputPath/frames_rgb_flow/flow/test/D3" "frames_rgb_flow/flow/test/D3/P22_02.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P22/P22_03.tar" "$outputPath/frames_rgb_flow/flow/test/D3" "frames_rgb_flow/flow/test/D3/P22_03.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P22/P22_04.tar" "$outputPath/frames_rgb_flow/flow/test/D3" "frames_rgb_flow/flow/test/D3/P22_04.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P01/P01_11.tar" "$outputPath/frames_rgb_flow/rgb/test/D2" "frames_rgb_flow/rgb/test/D2/P01_11.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P01/P01_12.tar" "$outputPath/frames_rgb_flow/rgb/test/D2" "frames_rgb_flow/rgb/test/D2/P01_12.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P01/P01_13.tar" "$outputPath/frames_rgb_flow/rgb/test/D2" "frames_rgb_flow/rgb/test/D2/P01_13.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P01/P01_14.tar" "$outputPath/frames_rgb_flow/rgb/test/D2" "frames_rgb_flow/rgb/test/D2/P01_14.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P01/P01_15.tar" "$outputPath/frames_rgb_flow/rgb/test/D2" "frames_rgb_flow/rgb/test/D2/P01_15.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P08/P08_09.tar" "$outputPath/frames_rgb_flow/rgb/test/D1" "frames_rgb_flow/rgb/test/D1/P08_09.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P08/P08_10.tar" "$outputPath/frames_rgb_flow/rgb/test/D1" "frames_rgb_flow/rgb/test/D1/P08_10.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P08/P08_14.tar" "$outputPath/frames_rgb_flow/rgb/test/D1" "frames_rgb_flow/rgb/test/D1/P08_14.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P08/P08_15.tar" "$outputPath/frames_rgb_flow/rgb/test/D1" "frames_rgb_flow/rgb/test/D1/P08_15.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P08/P08_16.tar" "$outputPath/frames_rgb_flow/rgb/test/D1" "frames_rgb_flow/rgb/test/D1/P08_16.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P08/P08_17.tar" "$outputPath/frames_rgb_flow/rgb/test/D1" "frames_rgb_flow/rgb/test/D1/P08_17.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P22/P22_01.tar" "$outputPath/frames_rgb_flow/rgb/test/D3" "frames_rgb_flow/rgb/test/D3/P22_01.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P22/P22_02.tar" "$outputPath/frames_rgb_flow/rgb/test/D3" "frames_rgb_flow/rgb/test/D3/P22_02.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P22/P22_03.tar" "$outputPath/frames_rgb_flow/rgb/test/D3" "frames_rgb_flow/rgb/test/D3/P22_03.tar"
epic_download "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P22/P22_04.tar" "$outputPath/frames_rgb_flow/rgb/test/D3" "frames_rgb_flow/rgb/test/D3/P22_04.tar"

echo -e "\n### ------------------------------------------------------- ###\n"
echo "### All done!"
echo -e "\n### ------------------------------------------------------- ###\n"
