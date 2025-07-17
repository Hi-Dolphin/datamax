import re
from typing import Dict, List, Optional, Union

from datamax.parser.base import BaseLife, MarkdownOutputVo
from datamax.utils.lifecycle_types import LifeType


class RecursiveCharacterTextSplitter:
    """
    Implementation based on LangChain's RecursiveCharacterTextSplitter
    Recursively splits text by character list, trying to keep semantically related code blocks together
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        is_separator_regex: bool = False,
    ):
        """
        Initialize text splitter
        
        Args:
            separators: List of separators, defaults to generic separators
            chunk_size: Maximum size of chunks
            chunk_overlap: Number of overlapping characters between chunks
            length_function: Function to calculate length
            is_separator_regex: Whether separators are regular expressions
        """
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return self._split_text(text, self._separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text"""
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = self._split_text_with_regex(text, _separator, self._chunk_overlap)

        _good_splits = []
        _separator = "" if separator is None else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def _split_text_with_regex(
        self, text: str, separator: str, chunk_overlap: int
    ) -> List[str]:
        """Split text using regular expressions"""
        if separator:
            if self._is_separator_regex:
                splits = re.split(separator, text)
            else:
                splits = text.split(separator)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge split text chunks"""
        separator_len = self._length_function(separator)
        docs = []
        current_doc: List[str] = []
        total = 0

        for d in splits:
            _len = self._length_function(d)
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size:
                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        """Join documents"""
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text


class CodeParser(BaseLife):
    """
    Code parser, implemented based on LangChain's text splitter
    Supports code splitting and parsing for multiple programming languages
    """

    # Supported programming languages and their specific separators
    LANGUAGE_SEPARATORS = {
        "python": [
            # Class and function definitions
            "\nclass ",
            "\ndef ",
            "\n\ndef ",
            "\n\nclass ",
            # Decorators
            "\n@",
            # Comment blocks
            '"""',
            "'''",
            # Import statements
            "\nfrom ",
            "\nimport ",
            # Control structures
            "\nif ",
            "\nfor ",
            "\nwhile ",
            "\nwith ",
            "\ntry:",
            "\nexcept ",
            # Generic separators
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "javascript": [
            # Function definitions
            "\nfunction ",
            "\nconst ",
            "\nlet ",
            "\nvar ",
            # Class definitions
            "\nclass ",
            # Comments
            "\n//",
            "\n/*",
            "*/",
            # Import/export statements
            "\nimport ",
            "\nexport ",
            # Control structures
            "\nif (",
            "\nfor (",
            "\nwhile (",
            "\ntry {",
            "\ncatch (",
            # Generic separators
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "java": [
            # Class and interface definitions
            "\npublic class ",
            "\nclass ",
            "\npublic interface ",
            "\ninterface ",
            # Method definitions
            "\npublic ",
            "\nprivate ",
            "\nprotected ",
            "\nstatic ",
            # Comments
            "\n//",
            "\n/*",
            "*/",
            # Import statements
            "\nimport ",
            "\npackage ",
            # Control structures
            "\nif (",
            "\nfor (",
            "\nwhile (",
            "\ntry {",
            "\ncatch (",
            # Generic separators
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "cpp": [
            # Include and namespace
            "\n#include ",
            "\nnamespace ",
            "\nusing namespace ",
            # Class and struct definitions
            "\nclass ",
            "\nstruct ",
            # Function definitions
            "\nint ",
            "\nvoid ",
            "\nstatic ",
            "\ninline ",
            # Comments
            "\n//",
            "\n/*",
            "*/",
            # Preprocessor directives
            "\n#",
            # Control structures
            "\nif (",
            "\nfor (",
            "\nwhile (",
            "\ntry {",
            "\ncatch (",
            # Generic separators
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "go": [
            # Package and import
            "\npackage ",
            "\nimport ",
            # Type definitions
            "\ntype ",
            "\nstruct {",
            "\ninterface {",
            # Function definitions
            "\nfunc ",
            # Variable definitions
            "\nvar ",
            "\nconst ",
            # Comments
            "\n//",
            "\n/*",
            "*/",
            # Control structures
            "\nif ",
            "\nfor ",
            "\nswitch ",
            "\nselect {",
            # Generic separators
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "rust": [
            # Module and use declarations
            "\nmod ",
            "\nuse ",
            # Struct and enum definitions
            "\nstruct ",
            "\nenum ",
            "\ntrait ",
            "\nimpl ",
            # Function definitions
            "\nfn ",
            "\npub fn ",
            # Variable definitions
            "\nlet ",
            "\nconst ",
            "\nstatic ",
            # Comments
            "\n//",
            "\n/*",
            "*/",
            # Control structures
            "\nif ",
            "\nfor ",
            "\nwhile ",
            "\nmatch ",
            # Generic separators
            "\n\n",
            "\n",
            " ",
            "",
        ],
    }

    def __init__(self, file_path: Union[str, list], domain: str = "Technology"):
        super().__init__(domain=domain)
        self.file_path = file_path

    @staticmethod
    def detect_language(file_path: str, content: str = None) -> str:
        """
        Detect programming language type
        
        Args:
            file_path: File path
            content: File content (optional)
            
        Returns:
            Detected programming language type
        """
        # Detection based on file extension
        extension_map = {
            ".py": "python",
            ".js": "javascript", 
            ".jsx": "javascript",
            ".ts": "javascript",
            ".tsx": "javascript",
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "cpp",
            ".h": "cpp",
            ".hpp": "cpp",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby",
            ".cs": "csharp",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
        }
        
        # Get file extension
        extension = ""
        if "." in file_path:
            extension = "." + file_path.split(".")[-1].lower()
        
        detected_lang = extension_map.get(extension, "generic")
        
        # If extension-based detection fails, further detect based on content
        if detected_lang == "generic" and content:
            if "def " in content and "import " in content:
                return "python"
            elif "function" in content and ("const " in content or "let " in content):
                return "javascript"
            elif "class " in content and "public " in content:
                return "java"
            elif "#include" in content and "int main" in content:
                return "cpp"
            elif "package " in content and "func " in content:
                return "go"
            elif "fn " in content and "let " in content:
                return "rust"
        
        return detected_lang

    @staticmethod
    def read_code_file(file_path: str) -> str:
        """
        Read code file content
        
        Args:
            file_path: Code file path
            
        Returns:
            File content string
        """
        try:
            # Try common encoding formats
            encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
            
            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
                    
            # If all encodings fail, use binary mode and try to decode
            with open(file_path, "rb") as file:
                raw_data = file.read()
                # Try to detect encoding using chardet
                try:
                    import chardet
                    detected = chardet.detect(raw_data)
                    if detected["encoding"]:
                        return raw_data.decode(detected["encoding"])
                except ImportError:
                    pass
                
                # Finally try utf-8 and ignore errors
                return raw_data.decode("utf-8", errors="ignore")
                
        except Exception as e:
            raise Exception(f"Cannot read file {file_path}: {str(e)}")

    def get_splitter_for_language(self, language: str, chunk_size: int = 4000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
        """
        Get corresponding text splitter based on programming language
        
        Args:
            language: Programming language type
            chunk_size: Chunk size
            chunk_overlap: Overlap size
            
        Returns:
            Configured text splitter
        """
        separators = self.LANGUAGE_SEPARATORS.get(language, ["\n\n", "\n", " ", ""])
        
        return RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_code(self, content: str, language: str, chunk_size: int = 4000, chunk_overlap: int = 200) -> List[str]:
        """
        Split code content
        
        Args:
            content: Code content
            language: Programming language type
            chunk_size: Chunk size
            chunk_overlap: Overlap size
            
        Returns:
            List of split code chunks
        """
        splitter = self.get_splitter_for_language(language, chunk_size, chunk_overlap)
        return splitter.split_text(content)

    def format_code_chunks_to_markdown(self, chunks: List[str], language: str, file_path: str) -> str:
        """
        Format code chunks to Markdown format
        
        Args:
            chunks: List of code chunks
            language: Programming language type
            file_path: Source file path
            
        Returns:
            Formatted Markdown content
        """
        markdown_content = f"# Code Analysis: {file_path}\n\n"
        markdown_content += f"**Language:** {language}\n"
        markdown_content += f"**Total Chunks:** {len(chunks)}\n\n"
        
        for i, chunk in enumerate(chunks, 1):
            markdown_content += f"## Chunk {i}\n\n"
            markdown_content += f"```{language}\n"
            markdown_content += chunk
            markdown_content += "\n```\n\n"
            
            # Add chunk statistics
            lines = chunk.count('\n') + 1
            chars = len(chunk)
            markdown_content += f"*Lines: {lines}, Characters: {chars}*\n\n"
            markdown_content += "---\n\n"
        
        return markdown_content

    def parse(self, file_path: str, chunk_size: int = 4000, chunk_overlap: int = 200) -> Dict:
        """
        Parse code file
        
        Args:
            file_path: Code file path
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap size
            
        Returns:
            Parse result dictionary
        """
        try:
            extension = self.get_file_extension(file_path)
            
            # 1) Start processing
            lc_start = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Code Analysis",
                life_type=LifeType.DATA_PROCESSING,
            )
            
            # 2) Read code file content
            content = self.read_code_file(file_path)
            
            # 3) Detect programming language
            language = self.detect_language(file_path, content)
            
            # 4) Split code
            chunks = self.split_code(content, language, chunk_size, chunk_overlap)
            
            # 5) Format to Markdown
            markdown_content = self.format_code_chunks_to_markdown(chunks, language, file_path)
            
            # 6) Construct output object and add start lifecycle
            output_vo = MarkdownOutputVo(extension, markdown_content)
            output_vo.add_lifecycle(lc_start)
            
            # 7) Processing completed
            lc_end = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Code Analysis",
                life_type=LifeType.DATA_PROCESSED,
            )
            output_vo.add_lifecycle(lc_end)
            
            return output_vo.to_dict()
            
        except Exception as e:
            # 8) Processing failed
            lc_fail = self.generate_lifecycle(
                source_file=file_path,
                domain=self.domain,
                usage_purpose="Code Analysis",
                life_type=LifeType.DATA_PROCESS_FAILED,
            )
            # Optional: Construct output object for failure case
            output_vo = MarkdownOutputVo(
                extension=self.get_file_extension(file_path),
                content=f"# Code Parsing Error\n\nFailed to parse {file_path}:\n\n```\n{str(e)}\n```"
            )
            output_vo.add_lifecycle(lc_fail)
            raise Exception(f"Code parsing failed: {str(e)}")

    def parse_multiple_files(self, file_paths: List[str], chunk_size: int = 4000, chunk_overlap: int = 200) -> List[Dict]:
        """
        Batch parse multiple code files
        
        Args:
            file_paths: List of code file paths
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap size
            
        Returns:
            List of parse results
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.parse(file_path, chunk_size, chunk_overlap)
                results.append(result)
            except Exception as e:
                # Log failed files but continue processing other files
                print(f"Warning: Failed to parse file {file_path}: {str(e)}")
                continue
                
        return results 