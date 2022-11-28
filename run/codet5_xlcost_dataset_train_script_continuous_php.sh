cd ../code
pretrained_model=Salesforce/codet5-base
train_size="1000"
array=("C++" "PHP" "Java" "PHP" "C#" "PHP" "JavaScript" "PHP" "Python" "PHP"
"PHP" "C++" "PHP" "Java" "PHP" "C#" "PHP" "JavaScript" "PHP" "Python"
)

for num in {0..19..2}
do
  first=${array[$num]}
  second=${array[$num+1]}
  output_dir=../models/xlcost_dataset_model/$first"2"$second"-"$train_size"train"
  echo $output_dir

  CUDA_VISIBLE_DEVICES=0,1 python run_codet5.py \
	--do_train \
	--do_eval \
	--do_test \
	--model_type codet5 \
	--model_name_or_path $pretrained_model \
	--train_filename ../data/XLCoST_parallel_6/$first-$train_size.train,../data/XLCoST_parallel_6/$second-$train_size.train \
	--dev_filename ../data/XLCoST_parallel_6/$first-few.test,../data/XLCoST_parallel_6/$second-few.test \
	--test_filename ../data/XLCoST_parallel_6/$first.test,../data/XLCoST_parallel_6/$second.test \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 4 \
	--eval_batch_size 4 \
	--train_steps 10000 \
	--eval_steps 5000
done