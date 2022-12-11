cd $(dirname $0)
cd ../code
#dataset_name=TransCoder_test
#supported_languages=("C++" "Java" "Python")

#dataset_name=CodeXGLUE_original
dataset_name=CodeXGLUE_tokenized
supported_languages=("C#" "Java")

#dataset_name=Linq2Stream
#dataset_name=Linq2Stream_tokenized
#supported_languages=("C#" "Java")

#dataset_name=XLCoST_parallel_5
#supported_languages=("C++" "Java" "C#" "JavaScript" "Python")

#dataset_name=XLCoST_parallel_6
#supported_languages=("C++" "Java" "C#" "JavaScript" "Python" "PHP")


for first in ${supported_languages[*]}; do
  for second in ${supported_languages[*]}; do
    if [ $first == $second ]; then
      continue
    fi
    output_dir=$dataset_name"_model"/$train_size"train"/$first"2"$second
    echo $output_dir
    python run_simple_copy.py \
    --test_filename ../data/$dataset_name/$first.test,../data/$dataset_name/$second.test
  done
done