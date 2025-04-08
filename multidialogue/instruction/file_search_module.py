import os
import json
import time
from typing import List, Dict, Any, Optional
import argparse
from openai import OpenAI
from tqdm import tqdm
import requests

class FileSearchModule:
    """
    A module for uploading files to OpenAI and searching content within them.
    Uses OpenAI's file search API to enable semantic search capabilities.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the file search module with an OpenAI API key.
        
        Args:
            api_key: OpenAI API key (will use environment variable if not provided)
        """
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.file_ids = []
        self.file_metadata = {}

    def list_files(self) -> List[Dict]:
        """
        List all files uploaded to the OpenAI API.
        
        Returns:
            List of file objects
        """
        try:
            response = self.client.files.list()
            return response.data
        except Exception as e:
            print(f"Error listing files: {str(e)}")
            return []

    def upload_file(self, file_path: str, purpose: str = "assistants") -> str:
        """
        Upload a file to the OpenAI API and add it to a vector store.
        
        Args:
            file_path: Path to the file to upload
            purpose: Purpose of the file ('assistants' for file search)
            
        Returns:
            File ID if successful, empty string otherwise
        """
        try:
            print(f"Uploading {file_path}...")
            with open(file_path, "rb") as file:
                response = self.client.files.create(
                    file=file,
                    purpose=purpose
                )
            
            file_id = response.id
            self.file_ids.append(file_id)
            self.file_metadata[file_id] = {
                "name": os.path.basename(file_path),
                "path": file_path,
                "upload_time": time.time()
            }
            
            print(f"File uploaded successfully with ID: {file_id}")
            
            # Try to add file to a vector store
            try:
                # List existing vector stores
                vector_stores = self.client.beta.vector_stores.list()
                
                # Use first vector store or create a new one
                vector_store_id = None
                if vector_stores.data:
                    vector_store_id = vector_stores.data[0].id
                    print(f"Using existing vector store: {vector_store_id}")
                else:
                    new_store = self.client.beta.vector_stores.create(
                        name="File Search Vector Store"
                    )
                    vector_store_id = new_store.id
                    print(f"Created new vector store: {vector_store_id}")
                
                # Add file to vector store
                if vector_store_id:
                    try:
                        self.client.beta.vector_stores.files.create(
                            vector_store_id=vector_store_id,
                            file_id=file_id
                        )
                        print(f"Added file to vector store {vector_store_id}")
                    except Exception as e:
                        print(f"Warning: Could not add file to vector store: {str(e)}")
            except Exception as e:
                print(f"Warning: Vector store operations failed: {str(e)}")
            
            return file_id
            
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            return ""

    def upload_directory(self, directory_path: str, file_types: List[str] = None, recursive: bool = True) -> List[str]:
        """
        Upload all files in a directory to the OpenAI API.
        
        Args:
            directory_path: Path to the directory
            file_types: List of file extensions to upload (e.g., ['.txt', '.pdf'])
            recursive: Whether to search subdirectories
            
        Returns:
            List of file IDs that were successfully uploaded
        """
        uploaded_file_ids = []
        
        if file_types is None:
            # Default to common text file types
            file_types = ['.txt', '.md', '.pdf', '.json', '.py', '.js', '.html', '.css', '.csv']
        
        for root, dirs, files in os.walk(directory_path):
            if not recursive and root != directory_path:
                continue
                
            for file in tqdm(files, desc=f"Uploading files in {root}"):
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_types and file_ext not in file_types:
                    continue
                
                file_id = self.upload_file(file_path)
                if file_id:
                    uploaded_file_ids.append(file_id)
            
        return uploaded_file_ids

    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from the OpenAI API.
        
        Args:
            file_id: ID of the file to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self.client.files.delete(file_id)
            
            if file_id in self.file_ids:
                self.file_ids.remove(file_id)
            
            if file_id in self.file_metadata:
                del self.file_metadata[file_id]
                
            print(f"File {file_id} deleted successfully")
            return True
            
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
            return False

    def delete_all_files(self) -> bool:
        """
        Delete all files uploaded through this module.
        
        Returns:
            True if all deletions were successful, False otherwise
        """
        success = True
        
        for file_id in self.file_ids.copy():
            if not self.delete_file(file_id):
                success = False
                
        return success

    def create_assistant(self, name: str, instructions: str, model: str = "gpt-4o") -> str:
        """
        Create an assistant with file search capabilities.
        
        Args:
            name: Name of the assistant
            instructions: Instructions for the assistant
            model: Model to use for the assistant
            
        Returns:
            Assistant ID if successful, empty string otherwise
        """
        try:
            # 创建助手
            assistant = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                model=model,
                tools=[{"type": "file_search"}]
            )
            
            # 使用REST API添加文件到助手
            for file_id in self.file_ids:
                try:
                    headers = {
                        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                        "Content-Type": "application/json",
                        "OpenAI-Beta": "assistants=v1"
                    }
                    
                    url = f"https://api.openai.com/v1/assistants/{assistant.id}/files"
                    payload = {"file_id": file_id}
                    
                    response = requests.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        print(f"Successfully added file {file_id} to assistant")
                    else:
                        print(f"Failed to add file {file_id}: {response.text}")
                except Exception as e:
                    print(f"Error adding file {file_id}: {str(e)}")
            
            print(f"Assistant created with ID: {assistant.id}")
            return assistant.id
            
        except Exception as e:
            print(f"Error creating assistant: {str(e)}")
            return ""

    def search_files(self, query: str, file_ids: List[str] = None, max_results: int = 10) -> List[Dict]:
        """
        Search files using the OpenAI API with assistants.
        
        Args:
            query: Search query
            file_ids: List of file IDs to search (defaults to all uploaded files)
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        if not file_ids:
            # Check if we have file IDs in memory
            if self.file_ids:
                file_ids = self.file_ids
            else:
                # Try to get all files from API
                files = self.list_files()
                if files:
                    file_ids = [file.id for file in files]
                else:
                    print("No files available for search. Please upload files first.")
                    return []
            
        if not file_ids:
            print("No files available for search. Please upload files first.")
            return []
        
        print(f"Searching in {len(file_ids)} files: {file_ids}")
        
        try:
            # 直接使用Beta Assistants API进行搜索 - 这个API不需要function参数
            print("Creating search assistant...")
            
            # 创建助手
            assistant = self.client.beta.assistants.create(
                name="File Search Assistant",
                instructions="You are a helpful assistant that searches through files and provides specific information based on user queries. Be thorough and cite your sources.",
                model="gpt-4o",
                tools=[{"type": "file_search"}]
            )
            
            print(f"Created assistant: {assistant.id}")
            
            # 向助手添加文件
            for file_id in file_ids:
                try:
                    # 使用客户端直接调用API
                    headers = {
                        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                        "Content-Type": "application/json",
                        "OpenAI-Beta": "assistants=v1"
                    }
                    
                    url = f"https://api.openai.com/v1/assistants/{assistant.id}/files"
                    payload = {"file_id": file_id}
                    
                    response = requests.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        print(f"Successfully added file {file_id} to assistant")
                    else:
                        print(f"Failed to add file {file_id}: {response.text}")
                except Exception as e:
                    print(f"Error adding file {file_id}: {str(e)}")
            
            # 创建线程
            thread = self.client.beta.threads.create()
            print(f"Created thread: {thread.id}")
            
            # 添加消息
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
            print(f"Added message to thread: {message.id}")
            
            # 运行助手
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            print(f"Started run: {run.id}")
            
            # 等待运行完成
            max_wait_time = 120  # 2分钟
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                
                status = run_status.status
                print(f"Run status: {status}")
                
                if status == "completed":
                    break
                elif status in ["failed", "cancelled", "expired"]:
                    print(f"Run failed with status: {status}")
                    # 尝试获取错误信息
                    if hasattr(run_status, "last_error"):
                        print(f"Error: {run_status.last_error}")
                    return []
                
                # 等待3秒后再次检查
                time.sleep(3)
            
            if time.time() - start_time >= max_wait_time:
                print("Search timed out")
                return []
            
            # 获取结果
            print("Retrieving search results...")
            messages = self.client.beta.threads.messages.list(
                thread_id=thread.id,
                order="desc"  # 获取最新消息
            )
            
            # 提取结果
            search_results = []
            for message in messages.data:
                if message.role == "assistant":
                    content_text = []
                    for content in message.content:
                        if content.type == "text":
                            content_text.append(content.text.value)
                    
                    if content_text:
                        search_results.append({
                            "content": "\n".join(content_text),
                            "message_id": message.id
                        })
            
            # 清理资源
            try:
                print(f"Cleaning up resources...")
                self.client.beta.assistants.delete(assistant.id)
            except Exception as e:
                print(f"Warning: Could not delete assistant: {str(e)}")
            
            return search_results[:max_results]
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def save_metadata(self, filepath: str) -> bool:
        """
        Save metadata about uploaded files to a JSON file.
        
        Args:
            filepath: Path to save the metadata file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metadata = {
                "file_ids": self.file_ids,
                "file_metadata": self.file_metadata
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            print(f"Metadata saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")
            return False

    def load_metadata(self, filepath: str) -> bool:
        """
        Load metadata about previously uploaded files from a JSON file.
        
        Args:
            filepath: Path to the metadata file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            self.file_ids = metadata.get("file_ids", [])
            self.file_metadata = metadata.get("file_metadata", {})
            
            print(f"Loaded metadata for {len(self.file_ids)} files")
            return True
            
        except Exception as e:
            print(f"Error loading metadata: {str(e)}")
            return False


def main():
    """Command-line interface for the file search module"""
    parser = argparse.ArgumentParser(description="Upload files and search content with OpenAI API")
    
    # Add metadata file argument to all commands
    parser.add_argument("--metadata", type=str, default="file_metadata.json",
                       help="Path to metadata file (default: file_metadata.json)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload file(s)")
    upload_parser.add_argument("--file", type=str, help="Path to a file to upload")
    upload_parser.add_argument("--directory", type=str, help="Path to a directory to upload")
    upload_parser.add_argument("--recursive", action="store_true", help="Recursively upload files in directory")
    upload_parser.add_argument("--file-types", type=str, nargs="+", help="File types to upload (e.g., .txt .pdf)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List uploaded files")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete file(s)")
    delete_parser.add_argument("--file-id", type=str, help="ID of the file to delete")
    delete_parser.add_argument("--all", action="store_true", help="Delete all files")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search files")
    search_parser.add_argument("--query", type=str, required=True, help="Search query")
    search_parser.add_argument("--file-ids", type=str, nargs="+", help="IDs of files to search")
    search_parser.add_argument("--max-results", type=int, default=10, help="Maximum number of results")
    
    # Save/load metadata
    save_parser = subparsers.add_parser("save", help="Save metadata")
    save_parser.add_argument("--file", type=str, required=True, help="Path to save metadata")
    
    load_parser = subparsers.add_parser("load", help="Load metadata")
    load_parser.add_argument("--file", type=str, required=True, help="Path to load metadata from")
    
    args = parser.parse_args()
    
    # Initialize module
    module = FileSearchModule()
    
    # Try to load metadata if file exists and command is not 'load'
    if args.command != "load" and os.path.exists(args.metadata):
        print(f"Loading metadata from {args.metadata}")
        module.load_metadata(args.metadata)
    
    if args.command == "upload":
        if args.file:
            module.upload_file(args.file)
        elif args.directory:
            file_types = args.file_types if args.file_types else None
            module.upload_directory(args.directory, file_types, args.recursive)
        
        # Save metadata after upload
        module.save_metadata(args.metadata)
            
    elif args.command == "list":
        files = module.list_files()
        if files:
            print("Uploaded files:")
            for file in files:
                print(f"ID: {file.id}, Name: {file.filename}, Purpose: {file.purpose}")
        else:
            print("No files found")
            
    elif args.command == "delete":
        if args.file_id:
            module.delete_file(args.file_id)
        elif args.all:
            if input("Are you sure you want to delete all files? (y/n): ").lower() == 'y':
                module.delete_all_files()
        
        # Save updated metadata after deletion
        module.save_metadata(args.metadata)
                
    elif args.command == "search":
        file_ids = args.file_ids if args.file_ids else None
        results = module.search_files(args.query, file_ids, args.max_results)
        
        if results:
            print("\nSearch Results:")
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(result["content"])
        else:
            print("No search results found")
            
    elif args.command == "save":
        module.save_metadata(args.file)
        
    elif args.command == "load":
        module.load_metadata(args.file)
        # Also save to default metadata file if different
        if args.file != args.metadata:
            module.save_metadata(args.metadata)

if __name__ == "__main__":
    main() 