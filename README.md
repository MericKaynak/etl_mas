# Multi-Agent System for ETL Processes

This repository contains the source code for the bachelor thesis project, "Design and Implementation of a Multi-Agent System for ETL Processes". The project introduces a novel, AI-driven approach to automate and simplify the extraction, transformation, and loading (ETL) of data from heterogeneous sources.

## Project Overview

The core of this project is a sophisticated multi-agent system that leverages Large Language Models (LLMs) to intelligently manage and execute complex data integration workflows. The system is designed to be user-friendly, allowing users to define their data integration requirements through an interactive chat interface. By using a standardized metamodel for the data schema (LinkML), the system ensures a high degree of automation and accuracy in the ETL process.

### Key Features

*   **Interactive Data Modeling:** Users can interactively define the target data model with the assistance of an AI agent, which generates a `LinkML` schema based on the user's input.
*   **Automated ETL Pipelines:** Once the data model is defined, the system autonomously orchestrates the entire ETL process, from data extraction to loading.
*   **Multi-Agent Architecture:** The system is built on a modular, multi-agent architecture, where specialized agents are responsible for specific tasks such as schema analysis, data engineering, and data analysis.
*   **Support for Heterogeneous Data Sources:** The system is designed to handle a variety of data formats, including CSV, JSON, and Excel.
*   **Flexible and Extensible:** The use of a standardized framework and a modular design makes the system easy to extend and adapt to new requirements.

## System Architecture

The architecture of the multi-agent system is designed to be modular and scalable. The following diagram illustrates the main components and their interactions:

![MAS](bild.svg)
