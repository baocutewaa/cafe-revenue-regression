# -*- coding: utf-8 -*-
"""
Script to generate Group2_Report.docx for Cafe Revenue Regression Project.
Based on Sample_Report.docx format, docs markdown reports, and notebook code.
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

print("Python Docx loaded successfully")
