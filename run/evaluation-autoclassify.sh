cd $(dirname $0)
dataset_name=TransCoder_autoclassify_test
model_name=transcoder_test_model
supported_train_sizes=("470" "200" "100")
supported_languages=("C++" "Java")
levels=("0" "1" "2" "3")

for first in ${supported_languages[*]}; do
  for second in ${supported_languages[*]}; do
    if [ $first == $second ]; then
      continue
    fi
    for train_size in ${supported_train_sizes[*]}; do
      output_dir=../result/TransCoder_autoclassify_test/$train_size"train"/$first"2"$second/01
      mkdir -p output_dir
      CUDA_VISIBLE_DEVICES=0 python ../code/run_codet5.py \
        --do_test \
        --model_type codet5 \
        --model_name_or_path Salesforce/codet5-base \
        --load_model_path ../models/$model_name/"$train_size"train/"$first"2"$second"/checkpoint-last/pytorch_model.bin \
        --dev_filename ../data/Transcoder_classify_auto/"$first".test,../data/Transcoder_classify_auto/"$second".test \
        --test_filename ../data/Transcoder_classify_auto/"$first"-L1.test,../data/Transcoder_classify_auto/"$second"-L1.test \
        --output_dir "$output_dir" \
        --max_source_length 512 \
        --max_target_length 512 \
        --beam_size 5 \
        --eval_batch_size 6
      output_dir=../result/TransCoder_autoclassify_test/$train_size"train"/$first"2"$second/23
      mkdir -p output_dir
      CUDA_VISIBLE_DEVICES=0 python ../code/run_codet5.py \
        --do_test \
        --model_type codet5 \
        --model_name_or_path Salesforce/codet5-base \
        --load_model_path ../models/$model_name/"$train_size"train/"$first"2"$second"/checkpoint-last/pytorch_model.bin \
        --dev_filename ../data/Transcoder_classify_auto/"$first"-L2.test,../data/Transcoder_classify_auto/"$second"-L2.test \
        --test_filename ../data/Transcoder_classify_auto/"$first"-L3.test,../data/Transcoder_classify_auto/"$second"-L3.test \
        --output_dir "$output_dir" \
        --max_source_length 512 \
        --max_target_length 512 \
        --beam_size 5 \
        --eval_batch_size 6
      done
    done
  done
done
