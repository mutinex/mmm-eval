import click
from click.testing import CliRunner

@click.command()
@click.option('--test-names', multiple=True, default=('test1', 'test2'))
def test_command(test_names):
    print(f"Type: {type(test_names)}")
    print(f"Value: {test_names}")
    return test_names

if __name__ == "__main__":
    runner = CliRunner()
    result = runner.invoke(test_command, ['--test-names', 'accuracy', 'cross_validation'])
    print(f"Output: {result.output}")
    print(f"Result: {result.return_value}")
    print(f"Type: {type(result.return_value)}") 