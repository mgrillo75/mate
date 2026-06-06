import unittest
from fastapi import Request
from fastapi.templating import Jinja2Templates
from pathlib import Path

class TestTemplatesRendering(unittest.TestCase):
    def test_templates_load_and_render(self):
        project_root = Path(__file__).parent.parent.parent
        templates_dir = project_root / "templates"
        self.assertTrue(templates_dir.exists(), f"Templates dir not found: {templates_dir}")
        
        templates = Jinja2Templates(directory=str(templates_dir))
        
        # Create a mock Starlette request
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "path": "/",
            "headers": [],
        }
        request = Request(scope)
        
        # Test rendering login.html
        response = templates.TemplateResponse(request, "login.html", {
            "google_enabled": False,
            "github_enabled": False,
            "oauth_error": "",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"login", response.body.lower())

if __name__ == '__main__':
    unittest.main()
