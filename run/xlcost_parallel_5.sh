cd $(dirname $0)
cd ../code
pretrained_model=Salesforce/codet5-base
#supported_train_sizes=("2000" "1000" "500" "200" "100")
#supported_languages=("C++" "Java" "C#" "JavaScript" "Python")
#supported_part=("C++" "Java" "C#" "JavaScript" "Python")
supported_train_sizes=("100")
supported_languages=("C++" "Java" "C#" "JavaScript" "Python")
supported_part=("JavaScript" "Python")
for train_size in ${supported_train_sizes[*]}; do
  for first in ${supported_part[*]}; do
    for second in ${supported_languages[*]}; do
#      if [ $first != "PHP" ] && [ $second != "PHP" ]; then
#        continue
#      fi
      if [ $first == $second ]; then
        continue
      fi
      output_dir=../models/xlcost_dataset_5_model/$train_size"train"/$first"2"$second
      echo $output_dir
      CUDA_VISIBLE_DEVICES=0 python run_codet5.py \
      --do_train \
      --do_eval \
      --do_test \
      --model_type codet5 \
      --model_name_or_path $pretrained_model \
      --train_filename ../data/XLCoST_parallel_5/$first-$train_size.train,../data/XLCoST_parallel_5/$second-$train_size.train \
      --dev_filename ../data/XLCoST_parallel_5/$first-few.test,../data/XLCoST_parallel_5/$second-few.test \
      --test_filename ../data/XLCoST_parallel_5/$first.test,../data/XLCoST_parallel_5/$second.test \
      --output_dir $output_dir \
      --max_source_length 512 \
      --max_target_length 512 \
      --beam_size 6 \
      --train_batch_size 6 \
      --eval_batch_size 6 \
      --train_steps 6500 \
      --eval_steps 6500
    done
  done
done
