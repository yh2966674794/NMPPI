import os
# os.environ["PATH"] = "D:\Anaconda\envs\Python38"
def run_func(description, ppi_path, pseq_path, vec_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, graph_only_train, 
            batch_size, epochs):
    os.system("python -u gnn_train.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --split_new={} \
            --split_mode={} \
            --train_valid_index_path={} \
            --use_lr_scheduler={} \
            --save_path={} \
            --graph_only_train={} \
            --batch_size={} \
            --epochs={} \
            ".format(description, ppi_path, pseq_path, vec_path, 
                    split_new, split_mode, train_valid_index_path,
                    use_lr_scheduler, save_path, graph_only_train, 
                    batch_size, epochs))

if __name__ == "__main__":
    description = "test_string_bfs"

    ppi_path = "./data/protein.actions.SHS27k.STRING.txt"
    # pemb_path = "embedding_string_dict.pkl"
    # pseq_path = "./data/protein.SHS27k.sequences.dictionary.tsv"

    # ppi_path = "./data/protein.actions.SHS148k.STRING.txt"
    pemb_path = "./data/embeddings_dgi.pkl"
    # pseq_path = "./data/protein.SHS148k.sequences.dictionary.tsv"

    # ppi_path = "./data/9606.protein.actions.all_connected.txt"
    # pseq_path = "./data/protein.STRING_all_connected.sequences.dictionary.tsv"
    vec_path = "./data/vec5_CTC.txt"
    n = 1
    for i in range(n):
        split_new = "True"
        split_mode = "dfs"
        train_valid_index_path = "./train_valid_index_json/string.random.fold1.json"

        use_lr_scheduler = "True"
        save_path = "./save_model/"
        graph_only_train = "False"

        batch_size = 2048
        epochs = 300

        run_func(description, ppi_path, pemb_path, vec_path,
                split_new, split_mode, train_valid_index_path,
                use_lr_scheduler, save_path, graph_only_train,
                batch_size, epochs)
