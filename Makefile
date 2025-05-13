format:
	uv run black .
	uv run isort .

run_app:
	make format
	uv run streamlit run main.py

update_prompts:
	uv run app/prompts/__init__.py
	
init_db:
	uv run app/orm/__init__.py