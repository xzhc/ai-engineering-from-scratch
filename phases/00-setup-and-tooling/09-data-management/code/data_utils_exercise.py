# 09-data-management's exercise
from dotenv import load_dotenv
from datasets import load_dataset
import time
import os

# import datasets

load_dotenv()

# help(datasets)
## 01 Load the 'glue' dataset with the 'mrpc' config and inspect the first five 5 examples.

# ds_builder = load_dataset_builder("nyu-mll/glue", "mrpc")
# print(ds_builder.info.download_size)

# dataset = load_dataset("nyu-mll/glue", "mrpc")
# # attrs = dir(dataset)
# # print(attrs)
# # print(type(dataset))
# train_ds = dataset["train"]
# # print(train_ds)
# # first_five_examples = train_ds[0:5]
# # print(first_five_examples)

# # print(type(first_five_examples))
# first_five_examples = train_ds[0:5]
# for i in range(len(first_five_examples["sentence1"])):
#     print(f"Example {i} sentence1: {first_five_examples['sentence1'][i]}")
#     print(f"Example {i} sentence2: {first_five_examples['sentence2'][i]}")
#     print(f"Example {i} label: {first_five_examples['label'][i]}")
#     print(f"Example {i} idx: {first_five_examples['idx'][i]}")

## 02 Stream the 'C4' dataset and count how many examples you can process in 10 seconds.
# c4_dataset_en = load_dataset("allenai/c4", "en", streaming=True)
# print(type(c4_dataset_en))
# print(c4_dataset_en)
# c4_train = c4_dataset_en["train"]


# count = 0
# start_time = time.perf_counter()
# time_limit = 10  # seconds

# for example in c4_train:
#     count += 1
#     elapsed_time = time.perf_counter() - start_time
#     if elapsed_time >= time_limit:
#         break

# total_time = time.perf_counter() - start_time
# print(f"Examples processed in {total_time:.2f} seconds: {count}")
# print(f"Throughtput: {count / total_time:.2f} examples/second")

## 03 Convert a dataset to Parquet and compare the file size to CSV.
# glue_dataset = load_dataset("nyu-mll/glue", "mrpc")
# glue_train = glue_dataset["train"]

# glue_train.to_csv("glue_train.csv")
# glue_train.to_parquet("glue_train.parquet")


# csv_size = os.path.getsize("glue_train.csv")
# parquet_size = os.path.getsize("glue_train.parquet")

# ratio = csv_size / parquet_size
# print(f"CSV size: {csv_size} bytes")
# print(f"Parquet size: {parquet_size} bytes")
# print(f"CSV is {ratio:.2f}x as larger as Parquet")


## 04 Create a 70/15/15 train/val/test split with a fixed seed and verify the sizes
SEED = 42
glue_dataset_train = load_dataset("nyu-mll/glue", "mrpc", split="train")

first_split = glue_dataset_train.train_test_split(test_size=0.7, seed=SEED)

train_dataset = first_split["train"]

temp_split = first_split["test"]
second_split = temp_split.train_test_split(test_size=0.5, seed=SEED)
val_dataset = second_split["train"]
test_dataset = second_split["test"]

total = len(glue_dataset_train)
print(f"Train: {len(train_dataset)} ({len(train_dataset)/total:.2%})")
print(f"Validation: {len(val_dataset)} ({len(val_dataset)/total:.2%})")
print(f"Test: {len(test_dataset)} ({len(test_dataset)/total:.2%})")

# Verify split sizes with visible output
split_sum = len(train_dataset) + len(val_dataset) + len(test_dataset)
print(f"Verification: {split_sum} == {total} → {split_sum == total}")
assert split_sum == total