from app import get_app


def main():
    app = get_app()
    app.run(debug=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
