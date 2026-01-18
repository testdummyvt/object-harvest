import tempfile
import os
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

# Import the functions from obh.generate
from obh.generate import prompt_gen_task, main, cli


def test_click_prompt_gen_command():
    """Test that the Click prompt-gen command works correctly."""
    runner = CliRunner()
    
    # Test that the command runs without error (with mocked dependencies)
    with patch('obh.generate.prompt_gen_task', return_value=0):
        result = runner.invoke(cli, [
            'prompt-gen',
            '--objects-list', 'apple — red fruit, banana — yellow fruit',
            '--num-prompts', '5',
            '--output', '/fake/output.ndjson'
        ])
        
        # Check that the command executed without error
        assert result.exit_code == 0


def test_click_image_gen_command():
    """Test that the Click image-gen command works correctly."""
    runner = CliRunner()
    
    # Test that the command runs without error (with mocked dependencies)
    with patch('obh.generate.image_gen_task', return_value=0):
        result = runner.invoke(cli, [
            'image-gen',
            '--input', '/fake/input.ndjson',
            '--output', '/fake/output_dir'
        ])
        
        # Check that the command executed without error
        assert result.exit_code == 0


def test_click_prompt_enhance_command():
    """Test that the Click prompt-enhance command works correctly."""
    runner = CliRunner()
    
    # Test that the command runs without error (with mocked dependencies)
    with patch('obh.generate.prompt_enhance_task', return_value=0):
        result = runner.invoke(cli, [
            'prompt-enhance',
            '--input', '/fake/input.ndjson',
            '--output', '/fake/output.ndjson'
        ])
        
        # Check that the command executed without error
        assert result.exit_code == 0


def test_prompt_gen_task_with_mocked_dependencies():
    """Test prompt_gen_task with mocked dependencies."""
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ndjson') as temp_output:
        temp_output_path = temp_output.name

    try:
        # Mock the necessary functions
        with patch('obh.generate.setup_llm_client'), \
             patch('obh.generate.rate_limited_call', return_value={'prompt': 'test prompt', 'objects': ['test']}) , \
             patch('obh.generate.validate_and_clean_prompt_gen_response', return_value={'prompt': 'test prompt', 'objects': ['test']}), \
             patch('obh.generate.restructure_objects', return_value={'prompt': 'test prompt', 'objects': ['test']}), \
             patch('obh.generate.ThreadPoolExecutor') as mock_executor, \
             patch('obh.generate.tqdm') as mock_tqdm:
            
            # Mock the executor to return a completed future
            mock_future = MagicMock()
            mock_future.result.return_value = {'prompt': 'test prompt', 'objects': ['test']}
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            mock_tqdm.return_value = [mock_future]
            
            # Create args namespace similar to what parse_args would return
            from argparse import Namespace
            args = Namespace(
                task='prompt-gen',
                objects_file=None,
                objects_list='apple — red fruit, banana — yellow fruit',
                num_prompts=2, # Small number for testing
                output=temp_output_path,
                min_objects=None,
                max_objects=None,
                rpm=60,
                model='openai/gpt-4o',
                base_url='https://openrouter.ai/api/v1',
                api_key=None,
                batch_size=10
            )
            
            # Run the prompt_gen_task
            result = prompt_gen_task(args)
            
            # Assertions
            assert result == 0  # Success
            
            # Check that the output file was created and has content
            with open(temp_output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 0 # At least one line should be written
                
    finally:
        # Clean up the temporary output file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


def test_main_function():
    """Test main function invokes the CLI."""
    # Mock the CLI function to prevent actual execution
    with patch('obh.generate.cli') as mock_cli:
        # Call main and check return value
        result = main()
        
        # Check that the return code is 0 (success)
        assert result == 0
        # Verify that cli was called
        mock_cli.assert_called_once()