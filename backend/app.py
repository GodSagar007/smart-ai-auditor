from backend.loader import load_model, load_data

def main():
    print("🔍 Smart AI Auditor - Initial Run")
    model = load_model()
    data = load_data()
    print("✅ Ready for audit! (This is the base setup)")

if __name__ == "__main__":
    main()
