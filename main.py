import os
import sys
import logging
import asyncio
import google.generativeai as genai  # type: ignore
from dotenv import load_dotenv  # type: ignore
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type  # type: ignore
import google.api_core.exceptions  # type: ignore
import unittest

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .envファイルから環境変数を読み込む
load_dotenv()

# Gemini APIキーの取得
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEYが環境変数に設定されていません。")

# Gemini APIの設定
genai.configure(api_key=GEMINI_API_KEY)

class GeminiModel:
    def __init__(self, model_name="gemini-1.5-pro"):
        self.model = genai.GenerativeModel(model_name=model_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        retry=retry_if_exception_type((google.api_core.exceptions.ResourceExhausted, asyncio.TimeoutError))
    )
    async def generate_test_code(self, prompt):
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            return self.extract_text_from_response(response)
        except google.api_core.exceptions.ResourceExhausted as e:
            logger.error("クォータを超過しました。使用状況を確認し、後でもう一度お試しください。")
            raise
        except Exception as e:
            logger.warning(f"エラーが発生しました: {e}。リトライします...")
            raise

    def extract_text_from_response(self, response):
        if hasattr(response, 'generations') and len(response.generations) > 0:
            return response.generations[0].text
        elif hasattr(response, 'text'):
            return response.text
        elif isinstance(response, dict):
            if 'generations' in response and isinstance(response['generations'], list) and len(response['generations']) > 0:
                return response['generations'][0].get('text', '')
            elif 'text' in response:
                return response.get('text', '')
        elif hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].text
        else:
            logger.error("レスポンスオブジェクトに生成されたテキストが見つかりません。")
            raise AttributeError("GenerateContentResponseに適切な属性が存在しません。")

    @staticmethod
    def sanitize_generated_text(text):
        # Markdownのコードブロックを削除
        text = text.replace("```python", "").replace("```", "").strip()
        return text

def extract_code_from_markdown(text):
    code_lines = []
    in_code_block = False
    for line in text.splitlines():
        if line.strip().startswith("```python"):
            in_code_block = True
            continue
        elif line.strip().startswith("```") and in_code_block:
            in_code_block = False
            continue
        if in_code_block:
            code_lines.append(line)
    return "\n".join(code_lines)

async def main():
    gemini_model = GeminiModel()

    # 機能要件1: src/にあるPythonファイルの洗い出し
    python_files = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'src')
    test_dir = os.path.join(script_dir, 'tests')

    if not os.path.exists(src_dir):
        logger.error(f"srcディレクトリが存在しません: {src_dir}")
        sys.exit(1)

    os.makedirs(test_dir, exist_ok=True)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':  # __init__.pyを無視
                python_files.append(os.path.join(root, file))

    if not python_files:
        logger.warning("srcディレクトリ内にPythonファイルが見つかりません。")
        sys.exit(0)

    logger.info("発見されたPythonファイル:")
    for file in python_files:
        logger.info(f" - {file}")

    # 機能要件3: テストスクリプトの構築
    test_files_created = []
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # Geminiを使ってテストコードを生成
        prompt = (
            "以下のPythonコードに対する100%網羅率の単体テストコードのみを生成してください。"
            "コメントや説明文は含めないでください。\n\n"
            "```python\n" +
            code +
            "\n```"
        )
        try:
            test_code = await gemini_model.generate_test_code(prompt)
            test_code = extract_code_from_markdown(test_code)

            # Log the generated test code for debugging
            logger.debug(f"Generated test code for {file_path}:\n{test_code}")

            if not test_code.strip():
                logger.error(f"生成されたテストコードが空です: {file_path}")
                continue

            # テストコードを保存
            test_file_name = f"test_{os.path.basename(file_path)}"
            test_file_path = os.path.join(test_dir, test_file_name)
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            logger.info(f"テストコードを生成しました: {test_file_path}")
            test_files_created.append(test_file_path)
        except Exception as e:
            logger.error(f"テストコードの生成中にエラーが発生しました: {file_path}。エラー: {e}")

    # 機能要件6: 単体テストを自動実行
    if test_files_created:
        logger.info("単体テストを実行します...")
        test_results = {}
        for test_file_path in test_files_created:
            # Use unittest to run the test file
            try:
                loader = unittest.TestLoader()
                suite = loader.discover(start_dir=test_dir, pattern=os.path.basename(test_file_path))
                runner = unittest.TextTestRunner()
                result = runner.run(suite)
                test_results[test_file_path] = result.wasSuccessful()
            except Exception as e:
                logger.error(f"テストの実行中にエラーが発生しました: {test_file_path}。エラー: {e}")
                test_results[test_file_path] = False

        # 機能要件7: 結果をレポート
        passed_tests = [k for k, v in test_results.items() if v]
        failed_tests = [k for k, v in test_results.items() if not v]

        logger.info("テスト結果:")
        logger.info(f"合格したテスト: {len(passed_tests)}")
        for test in passed_tests:
            logger.info(f" - {test}")
        logger.info(f"不合格のテスト: {len(failed_tests)}")
        for test in failed_tests:
            logger.info(f" - {test}")

        # 機能要件8: 失敗したテストを自動改修（ユーザー承認）
        if failed_tests:
            approval = input("失敗したスクリプトを自動修正しますか？ (yes/no): ")
            if approval.lower() == 'yes':
                for test_file in failed_tests:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        test_code = f.read()

                    prompt = (
                        "以下のテストコードは失敗しています。原因を特定し、元のコードとテストコードを修正してください。\n\n"
                        "```python\n" +
                        test_code +
                        "\n```"
                    )
                    try:
                        fixed_code = await gemini_model.generate_test_code(prompt)
                        fixed_code = extract_code_from_markdown(fixed_code)

                        # 生成されたコードのサニタイズ
                        fixed_code = gemini_model.sanitize_generated_text(fixed_code)

                        # 生成された修正コードが有効なPythonコードか確認
                        try:
                            compile(fixed_code, '<string>', 'exec')
                        except SyntaxError as e:
                            logger.error(f"修正後のコードに構文エラーがあります: {test_file}。エラー: {e}")
                            continue

                        # 修正されたコードを上書き
                        with open(test_file, 'w', encoding='utf-8') as f:
                            f.write(fixed_code)
                        logger.info(f"修正しました: {test_file}")
                    except Exception as e:
                        logger.error(f"テストコードの修正に失敗しました: {test_file}。エラー: {e}")
                        continue

    logger.info("処理が完了しました。")

if __name__ == "__main__":
    asyncio.run(main())