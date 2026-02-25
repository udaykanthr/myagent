import pytest
from multi_agent_coder.diff_display import _detect_hazards, HAZARD_WARN
from multi_agent_coder.config import Config

def test_detect_hazards_shrinkage():
    # File shrinkage > 50%
    old_content = "a" * 200
    new_content = "a" * 50
    hazards = _detect_hazards("any_file.txt", old_content, new_content)
    assert len(hazards) == 1
    assert hazards[0][0] == HAZARD_WARN
    assert "Significant size reduction" in hazards[0][1]

def test_detect_hazards_no_shrinkage():
    # Small shrinkage is fine
    old_content = "a" * 200
    new_content = "a" * 150
    hazards = _detect_hazards("any_file.txt", old_content, new_content)
    assert len(hazards) == 0

def test_detect_hazards_small_file_ignored():
    # Small files ignored
    old_content = "a" * 50
    new_content = "a" * 10
    hazards = _detect_hazards("any_file.txt", old_content, new_content)
    assert len(hazards) == 0

# def test_detect_hazards_package_json_dependency_loss():
#     # Dependencies removed
#     old_content = '{"dependencies": {"react": "^18.0.0"}}'
#     new_content = '{"name": "my-app"}'
#     hazards = _detect_hazards("package.json", old_content, new_content)
#     # Checks for dependency loss heuristics
#     assert any("Critical section 'dependencies' removed" in h[1] for h in hazards)

def test_detect_hazards_package_json_dependency_modification():
    # Simulating a manual edit to dependencies (even if not removed, generic warning for package.json)
    # The current logic flags *any* edit to package.json dependencies if they are present in new content?
    # Let's check the code:
    # "if '"dependencies"' in new_content or '"devDependencies"' in new_content:" -> WARN
    
    old_content = '{"name": "app", "dependencies": {}}'
    new_content = '{"name": "app", "dependencies": {"react": "^18"}}'
    
    # This should trigger the warning about manual dependency edits
    hazards = _detect_hazards("package.json", old_content, new_content)
    assert any("Verify this is not a manual dependency edit" in h[1] for h in hazards)

def test_detect_hazards_package_json_safe_edit():
    # Editing scripts is "safer" (though currently our logic might still flag if deps are present elsehwere)
    # If we only touch scripts and deps are NOT in the file (unlikely for real package.json) -> no warning
    # But if deps are in the file, and we output the full file including deps -> the warning "Verify..." might trigger 
    # if we implemented the check strictly.
    # In the current implementation: 
    #   if '"dependencies"' in new_content ... -> WARN
    # So actually, ANY edit to a package.json with dependencies will trigger the "Verify" warning.
    # This is consistent with "strict blocking/warning".
    
    old_content = '{"scripts": {"test": "jest"}}'
    new_content = '{"scripts": {"test": "vitest"}}'
    hazards = _detect_hazards("package.json", old_content, new_content)
    # No dependencies in file -> no dependency warning
    assert len(hazards) == 0
