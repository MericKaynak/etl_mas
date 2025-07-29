import json
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import os
import math
import streamlit as st

class SchemaAnalyzer:
    """
    Universal schema analyzer for JSON, CSV, Excel, and XML files.
    
    Automatically detects the file type and analyzes its schema,
    including support for nested arrays and objects via recursive introspection.

    Parameters:
        file_path (str): Path to the file to be analyzed.
        rows_per_sample_section (int): Number of rows to sample from head, middle, and tail for schema inference.
    """

    def __init__(self, file_path, rows_per_sample_section=20):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.file_path = file_path
        self.sample_chunk_size = rows_per_sample_section
        self.extension = os.path.splitext(file_path)[1].lower()

    def analyze(self):
        """
        Automatically detects the file format and analyzes its schema.

        Returns:
            dict: A dictionary representing the inferred schema or an error message.
        """
        try:
            if self.extension == ".json":
                return self._analyze_json()
            elif self.extension == ".csv":
                return self._analyze_csv()
            elif self.extension in [".xls", ".xlsx"]:
                return self._analyze_excel()
            elif self.extension == ".xml":
                return self._analyze_xml()
            else:
                return {"error": f"Unsupported file type: {self.extension}"}
        except Exception as e:
            import traceback
            return {"error": f"An error occurred: {e}", "traceback": traceback.format_exc()}

    def _determine_type(self, value):
        """
        Determines the type of a value recursively, including arrays and nested objects.

        Args:
            value: The value to inspect.

        Returns:
            str or dict or list: Inferred data type, or a recursive schema for objects and arrays.
        """
        if isinstance(value, dict):
            return self._analyze_schema(value)
        elif isinstance(value, list):
            if not value:
                return ["array of", "empty"]
            element_types = [self._determine_type(item) for item in value]
            merged_types = self._merge_types(element_types)
            return ["array of"] + merged_types
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, bool):
            return "boolean"
        elif value is None:
            return "null"
        else:
            return "unknown"

    def _infer_value_type(self, value: str):
        """
        Infers the data type from a string (used for CSV/Excel parsing).

        Args:
            value (str): The string to infer from.

        Returns:
            str: Inferred data type.
        """
        if not isinstance(value, str) or value.strip() == '':
            return self._determine_type(value)
        try: int(value); return "integer"
        except (ValueError, TypeError): pass
        try: float(value); return "number"
        except (ValueError, TypeError): pass
        if value.lower() in ['true', 'false']: return "boolean"
        return "string"

    def _merge_schemas(self, list_of_schemas):
        """
        Merges a list of schema dictionaries into a single schema.

        Args:
            list_of_schemas (list): List of schema dictionaries.

        Returns:
            dict: Merged schema.
        """
        if not list_of_schemas:
            return {}
        all_keys = set()
        for schema in list_of_schemas:
            if isinstance(schema, dict):
                all_keys.update(schema.keys())

        merged_schema = {}
        for key in all_keys:
            types_for_key = []
            for schema in list_of_schemas:
                if isinstance(schema, dict) and key in schema:
                    types_for_key.append(schema[key])
            merged_schema[key] = self._merge_types(types_for_key)[0] if len(self._merge_types(types_for_key)) == 1 else self._merge_types(types_for_key)
        return merged_schema

    def _merge_types(self, types):
        """
        Merges a list of data types (strings, dicts, lists) into a combined type.

        Args:
            types (list): List of inferred types.

        Returns:
            list: Merged and simplified type list.
        """
        simple_types = set()
        schemas = []

        for t in types:
            if isinstance(t, dict):
                schemas.append(t)
            elif isinstance(t, list) and t and t[0] == "array of":
                simple_types.add("array")
            elif isinstance(t, list):
                for inner_t in t:
                    if isinstance(inner_t, dict):
                        schemas.append(inner_t)
                    else:
                        simple_types.add(inner_t)
            elif t is not None:
                simple_types.add(t)

        final_types = sorted(list(simple_types))

        if schemas:
            merged_schema = self._merge_schemas(schemas)
            final_types.append(merged_schema)

        if not final_types:
            return ["unknown"]

        return final_types

    def _analyze_schema(self, data):
        """
        Analyzes a dictionary object and infers its schema.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Inferred schema.
        """
        if not isinstance(data, dict):
            return self._determine_type(data)
        schema = {}
        for key, value in data.items():
            schema[key] = self._determine_type(value)
        return schema

    def get_file_snippets(self, n):
        """
        Reads the first, middle, and last n lines of a file.

        Args:
            n (int): Number of lines to extract from each section.

        Returns:
            dict: Contains 'head', 'middle', and 'tail' string contents of the file.
        """
        try:
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(self.file_path, 'r', encoding='latin1') as f:
                    lines = f.readlines()

            total_lines = len(lines)
            head_lines = lines[:n]
            tail_lines = lines[-n:]
            middle_lines = lines if total_lines <= n else lines[max(0, (total_lines - n) // 2) : max(0, (total_lines - n) // 2) + n]

            return {
                "head": "".join(head_lines),
                "middle": "".join(middle_lines),
                "tail": "".join(tail_lines)
            }
        except Exception as e:
            import traceback
            return {
                "error": f"Failed to read file: {e}",
                "traceback": traceback.format_exc()
            }

    def _get_strategic_samples(self, all_rows):
        """
        Collects head, middle, and tail samples from a list of rows.

        Args:
            all_rows (list): Full list of row dictionaries.

        Returns:
            list: Unique representative sample rows.
        """
        total_rows = len(all_rows)
        chunk_size = self.sample_chunk_size
        if total_rows <= chunk_size * 2:
            return all_rows

        head = all_rows[:chunk_size]
        middle_start = max(chunk_size, (total_rows - chunk_size) // 2)
        middle = all_rows[middle_start : middle_start + chunk_size]
        tail = all_rows[-chunk_size:]
        samples = head + middle + tail

        seen_representations = set()
        unique_samples = []

        for row in samples:
            try:
                representation = json.dumps(row, sort_keys=True)
                if representation not in seen_representations:
                    seen_representations.add(representation)
                    unique_samples.append(row)
            except TypeError:
                rep_str = str(row)
                if rep_str not in seen_representations:
                    seen_representations.add(rep_str)
                    unique_samples.append(row)

        return unique_samples

    def _analyze_json(self):
        """
        Analyzes a JSON file and returns its schema.

        Returns:
            dict: Inferred schema.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self._determine_type(data)

    def _analyze_combined_samples(self, samples, use_infer_type=False):
        """
        Analyzes a list of sample rows and infers the overall schema.

        Args:
            samples (list): Sample rows from a CSV/Excel file.
            use_infer_type (bool): Reserved for future type inference extensions.

        Returns:
            dict: Inferred schema.
        """
        if not samples:
            return {"info": "No data samples found to analyze."}
        return self._determine_type(samples)

    def _analyze_csv(self):
        """
        Parses and analyzes a CSV file.

        Returns:
            dict: Inferred schema.
        """
        try:
            df = pd.read_csv(self.file_path, low_memory=False, keep_default_na=False)
        except Exception:
            df = pd.read_csv(self.file_path, low_memory=False, keep_default_na=False, encoding='latin1')

        if df.empty:
            return {"info": "CSV is empty or header-only"}

        rows = df.to_dict(orient='records')
        typed_rows = [{k: self._infer_value_type(v) if isinstance(v, str) else v for k, v in row.items()} for row in rows]

        samples = self._get_strategic_samples(typed_rows)
        return self._analyze_combined_samples(samples)

    def _analyze_excel(self):
        """
        Parses and analyzes an Excel file.

        Returns:
            dict: Inferred schema.
        """
        df = pd.read_excel(self.file_path, keep_default_na=False)
        if df.empty:
            return {"info": "Excel sheet is empty"}
        df = df.astype(str)
        rows = df.to_dict(orient='records')
        typed_rows = [{k: self._infer_value_type(v) for k, v in row.items()} for row in rows]

        samples = self._get_strategic_samples(typed_rows)
        return self._analyze_combined_samples(samples)

    def _analyze_xml(self):
        """
        Parses and analyzes an XML file.

        Returns:
            dict: Inferred schema.
        """
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        def xml_to_dict(elem):
            if not list(elem) and elem.text is None and elem.attrib:
                return self._analyze_schema(elem.attrib)
            if not list(elem):
                return self._infer_value_type(elem.text) if elem.text else None
            d = {}
            for child in list(elem):
                child_data = xml_to_dict(child)
                if child.tag in d:
                    if not isinstance(d[child.tag], list):
                        d[child.tag] = [d[child.tag]]
                    d[child.tag].append(child_data)
                else:
                    d[child.tag] = child_data
            return d

        dict_data = {root.tag: xml_to_dict(root)}
        return self._analyze_schema(dict_data)
