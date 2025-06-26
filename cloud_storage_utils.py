from google.cloud.storage import Client, transfer_manager
import os


def download_many_blobs_with_transfer_manager(
    bucket_name, gcs_folder_prefix, destination_directory="", workers=8
):
    storage_client = Client("tim-project")
    bucket = storage_client.bucket(bucket_name)

    if gcs_folder_prefix and not gcs_folder_prefix.endswith("/"):
        gcs_folder_prefix += "/"
    blob_names = list(
        map(
            lambda y: y.name,  # Get string name
            filter(
                lambda x: x.name[len(gcs_folder_prefix) :]
                and not x.name.endswith("/"),  # Filter so that we only have files
                bucket.list_blobs(prefix=gcs_folder_prefix),
            ),
        )
    )

    results = transfer_manager.download_many_to_path(
        bucket,
        blob_names,
        destination_directory=destination_directory,
        max_workers=workers,
        worker_type=transfer_manager.THREAD,
    )
    all_downloaded = True
    for name, result in zip(blob_names, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.
        if isinstance(result, Exception):
            print("Failed to download {} due to exception: {}".format(name, result))
            all_downloaded = False
    if all_downloaded:
        print(f"Completed download of gs://{bucket_name}/{gcs_folder_prefix}")


def list_local_files_os_walk(local_folder_path):
    """
    Recursively lists the full paths of all files within a local folder.

    Args:
        local_folder_path (str): The path to the local folder.

    Returns:
        list: A list of strings, where each string is the absolute path
              to a file within the specified folder and its subfolders.
              Returns an empty list if the path is invalid or not a directory.
    """
    file_paths = []
    # Ensure the provided path exists and is a directory
    if not os.path.isdir(local_folder_path):
        print(f"Error: Path '{local_folder_path}' is not a valid directory.")
        return []

    # os.walk yields (current_directory_path, list_of_subdirs, list_of_files)
    for root, dirs, files in os.walk(local_folder_path):
        for filename in files:
            # Construct the full path by joining the root directory and filename
            full_path = os.path.join(root, filename)
            file_paths.append(full_path)
    return file_paths


def upload_many_blobs_with_transfer_manager(
    bucket_name, source_directory="", workers=8
):
    storage_client = Client("tim-project")
    bucket = storage_client.bucket(bucket_name)
    filenames = list_local_files_os_walk(source_directory)
    results = transfer_manager.upload_many_from_filenames(
        bucket, filenames, max_workers=workers, worker_type=transfer_manager.THREAD
    )
    all_uploaded = True
    for name, result in zip(filenames, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
            all_uploaded = False
    if all_uploaded:
        print(f"Completed upload of {source_directory}")
