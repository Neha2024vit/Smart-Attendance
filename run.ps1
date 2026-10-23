$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$env:PYTHONPATH = $root
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
