# Fabric notebook source


# CELL ********************

# -*- coding: utf-8 -*-
"""Layer-1 unit tests for nb_00_shared_utils — the pure, stdlib-only helpers that
both the MMV and ECV bronze notebooks depend on (via %run in Fabric).

These run OFF Fabric with no third-party packages (no pyspark / pandas / pytest):
    python -m unittest discover -s fabric/tests -v
or from inside fabric/:
    python -m unittest tests.test_shared_utils -v

The Spark notebooks (nb_*_01/02/03) execute top-level `spark.*` / driver downloads
and cannot be imported off-Fabric; they are covered by Layer 2/3 (see README_tests.md).
nb_00_shared_utils.py, by contrast, is side-effect-free and imports cleanly here.
"""
import os
import sys
import unittest

# Make the parent `fabric/` dir importable so we can load nb_00_shared_utils.py.
FABRIC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FABRIC_DIR not in sys.path:
    sys.path.insert(0, FABRIC_DIR)

import nb_00_shared_utils as u  # noqa: E402


class TestTextHelpers(unittest.TestCase):
    def test_strip_accents(self):
        self.assertEqual(u.strip_accents("Cámara"), "Camara")
        self.assertEqual(u.strip_accents("ATLÁNTICO"), "ATLANTICO")
        self.assertEqual(u.strip_accents(""), "")

    def test_norm_key(self):
        # accent-strip + lower + drop non-alphanumeric
        self.assertEqual(u.norm_key("Código Departamento"), "codigodepartamento")
        self.assertEqual(u.norm_key("Total Votos"), "totalvotos")
        self.assertEqual(u.norm_key(None), "")

    def test_norm_name_postmortem_s7_fix(self):
        # The §7 fix: spaces and underscores are equivalent.
        self.assertEqual(u.norm_name("SAN ANDRÉS"), u.norm_name("SAN_ANDRES"))
        self.assertEqual(u.norm_name("LA GUAJIRA"), u.norm_name("LA_GUAJIRA"))
        self.assertEqual(u.norm_name(" antioquia "), "ANTIOQUIA")
        self.assertEqual(u.norm_name("LA__GUAJIRA"), "LA GUAJIRA")  # collapse repeats

    def test_slugify(self):
        self.assertEqual(u.slugify("Cámara de Representantes"), "CAMARA_DE_REPRESENTANTES")
        self.assertEqual(u.slugify(""), "SIN_CORPORACION")
        self.assertEqual(u.slugify("Asamblea"), "ASAMBLEA")


class TestParamParsing(unittest.TestCase):
    def test_parse_departments_all(self):
        req, keep = u.parse_departments_param("ALL")
        self.assertEqual(req, "ALL")
        self.assertIsNone(keep)

    def test_parse_departments_list(self):
        req, keep = u.parse_departments_param("Antioquia, Atlántico")
        self.assertEqual(req, ["Antioquia", "Atlántico"])
        # keep-set is normalized for name routing
        self.assertEqual(keep, {"ANTIOQUIA", "ATLANTICO"})

    def test_parse_years_string_and_list(self):
        self.assertEqual(u.parse_years_param("2023,2022,2019,2018"), [2023, 2022, 2019, 2018])
        self.assertEqual(u.parse_years_param([2018, 2019]), [2018, 2019])
        self.assertEqual(u.parse_years_param("2023, ,2022"), [2023, 2022])  # tolerates blanks


class TestEncodingCascade(unittest.TestCase):
    def test_plain_utf8(self):
        text, enc = u.decode_bytes("ANTIOQUIA;NIÑO".encode("utf-8"))
        self.assertEqual(text, "ANTIOQUIA;NIÑO")
        self.assertTrue(enc.startswith("utf-8"))

    def test_utf16_bom(self):
        text, enc = u.decode_bytes("HOLA MUNDO".encode("utf-16"))  # adds a BOM
        self.assertEqual(text, "HOLA MUNDO")
        self.assertIn("utf-16", enc)

    def test_utf8_sig_bom(self):
        text, enc = u.decode_bytes("CÓDIGO".encode("utf-8-sig"))
        self.assertEqual(text, "CÓDIGO")
        self.assertIn("utf-8-sig", enc)

    def test_lone_d1_n_tilde_fixup(self):
        # latin-1-style lone 0xD1 (Ñ) embedded in otherwise-utf-8 bytes.
        raw = b"MUNICIPIO DE NI\xd1O"
        text, enc = u.decode_bytes(raw)
        self.assertEqual(text, "MUNICIPIO DE NIÑO")
        self.assertIn("0xD1", enc)

    def test_latin1_fallback(self):
        raw = "Bogotá".encode("latin-1")  # 0xe1, invalid utf-8, no 0xD1
        text, enc = u.decode_bytes(raw)
        self.assertEqual(text, "Bogotá")
        self.assertEqual(enc, "latin-1")

    def test_no_replacement_chars_ever(self):
        for raw in (b"a,b,c", "Ñoño".encode("utf-8"), "x".encode("utf-16")):
            text, _ = u.decode_bytes(raw)
            self.assertNotIn("�", text)


class TestDelimiterDetection(unittest.TestCase):
    def test_semicolon(self):
        text = "a;b;c\n1;2;3\n4;5;6"
        self.assertEqual(u.detect_delimiter(text), ";")

    def test_comma(self):
        text = "a,b,c\n1,2,3\n4,5,6"
        self.assertEqual(u.detect_delimiter(text), ",")


class TestHeaderMap(unittest.TestCase):
    def test_header_present_maps_by_name(self):
        present, width, names = u.build_header_map(
            ["Codigo Departamento", "Nombre Departamento", "Total Votos"]
        )
        self.assertTrue(present)
        self.assertEqual(width, 3)
        self.assertEqual(names, ["codigo_departamento", "departamento", "total_votos"])

    def test_headerless_falls_back_to_schema_order(self):
        present, width, names = u.build_header_map(["01", "ANTIOQUIA", "005"])
        self.assertFalse(present)
        self.assertEqual(names, u.SCHEMA[:3])

    def test_abbreviated_presidential_aliases(self):
        # national Presidencial CSV vocabulary (dep/munNombre/votos ...)
        present, _, names = u.build_header_map(["DEP", "MunNombre", "VOTOS"])
        self.assertTrue(present)
        self.assertEqual(names, ["codigo_departamento", "municipio", "total_votos"])


class TestFileStem(unittest.TestCase):
    def test_national_presidential(self):
        self.assertEqual(u.file_stem(2018, "national", "1v", "PRESIDENTE"), "2018_PRESIDENTE_1V")
        self.assertEqual(u.file_stem(2022, "national", "", "PRESIDENTE"), "2022_PRESIDENTE")

    def test_department(self):
        self.assertEqual(
            u.file_stem(2023, "department", "", "Asamblea de Antioquia"),
            "2023_ASAMBLEA_DE_ANTIOQUIA",
        )


class TestSchemaInvariants(unittest.TestCase):
    def test_canonical_schema_width(self):
        self.assertEqual(u.N_FIELDS, 18)
        self.assertEqual(len(u.SCHEMA), 18)
        self.assertEqual(u.EXTRA_COLS, ["year", "election", "scope", "corporation"])

    def test_ecv_sources_have_required_keys(self):
        for year, cfg in u.ECV_SOURCES.items():
            self.assertIn("microdata", cfg, f"{year} missing microdata")
            self.assertIn("year_column", cfg, f"{year} missing year_column")


if __name__ == "__main__":
    unittest.main(verbosity=2)
