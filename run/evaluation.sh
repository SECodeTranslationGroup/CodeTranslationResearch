cd $(dirname $0)
#dataset_name=XLCoST_parallel_6
#model_name=xlcost_dataset_model
#supported_train_sizes=("1000" "500" "200" "100")
#supported_languages=("C++" "Java" "C#" "JavaScript" "Python" "PHP")

#dataset_name=Linq2Stream
#model_name=linq2stream_model
#supported_train_sizes=("500" "200" "100")
#supported_languages=("Java" "C#")

#dataset_name=CodeXGLUE_original
#model_name=codexglue_original_model
#supported_train_sizes=("10295" "1000" "500" "200" "100")
#supported_languages=("Java" "C#")

#dataset_name=TransCoder_test
#model_name=transcoder_test_model
#supported_train_sizes=("470" "200" "100")
#supported_languages=("Java" "C++" "Python")

dataset_name=XLCoST_parallel_5
model_name=xlcost_dataset_5_model
supported_train_sizes=("2000" "1000" "500" "200" "100")
supported_languages=("C++" "Java" "C#" "JavaScript" "Python")


stat_name=../result/$model_name/stat.txt
mkdir ../result/$model_name/
mkdir ../result/$model_name/output

for first in ${supported_languages[*]}; do
  for second in ${supported_languages[*]}; do
    for train_size in ${supported_train_sizes[*]}; do
      if [ $first == $second ]; then
        continue
      fi
      name=$first"2"$second"-"$train_size
      echo $output_dir
      python ../evaluator/my_eval/evaluate.py \
        --statistics_file $stat_name \
        --name $name \
        --output_file ../result/$model_name/output/$name"train.txt" \
        --source ../data/$dataset_name/$first".test" \
        --references ../models/$model_name/$train_size"train"/$first"2"$second/"test_1.gold" \
        --predictions ../models/$model_name/$train_size"train"/$first"2"$second/"test_1.output"
    done
  done
done
