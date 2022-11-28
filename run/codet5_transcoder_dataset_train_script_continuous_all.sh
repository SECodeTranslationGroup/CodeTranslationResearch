cd ../code
pretrained_model=Salesforce/codet5-base

array=("cpp" "java" "cpp" "py" "java" "cpp" "java" "py" "py" "cpp" "py" "java")
for num in {0..11..2}
do
  first=${array[$num]}
  second=${array[$num+1]}
  output_dir=../models/TransCoder_test_model/$first"2"$second"-200train"
  echo $output_dir

  CUDA_VISIBLE_DEVICES=0,1 python run_codet5.py \
	--do_train \
	--do_eval \
	--do_test \
	--model_type codet5 \
	--model_name_or_path $pretrained_model \
	--train_filename ../data/transcoder/train-200.$first,../data/transcoder/train-200.$second \
	--dev_filename ../data/transcoder/test-few.$first,../data/transcoder/test-few.$second \
	--test_filename ../data/transcoder/test.$first,../data/transcoder/test.$second \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 4 \
	--eval_batch_size 4 \
	--train_steps 10000 \
	--eval_steps 5000
done