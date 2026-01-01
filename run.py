from kfold_validation import validate
from model_use.main import choose_model
from plot import plot_training_history, plot_subject_dependet
import sys
import numpy as np


def run_experiment(model_name, emotion, category, k, validation_type='subject_independent', num_people=23):
    """
    اجرای آزمایش با دو حالت subject-independent یا subject-dependent
    
    Parameters:
    -----------
    model_name : str
        نام مدل (simpleNN, cnn_45138, capsnet2020, hippoLegS1)
    emotion : str
        نوع احساس (valence یا dominance)
    category : str
        دسته‌بندی (binary یا 5category)
    k : int
        تعداد fold برای cross-validation
    validation_type : str
        نوع اعتبارسنجی ('subject_independent' یا 'subject_dependent')
    num_people : int
        تعداد افراد (پیش‌فرض: 23)
    
    Returns:
    --------
    dict : نتایج آزمایش
    """
    
    if validation_type == 'subject_independent':
        # حالت Subject-Independent (مستقل از فرد)
        print(f"Running Subject-Independent validation with {k}-fold cross-validation...")
        print(f"Model: {model_name}, Emotion: {emotion}, Category: {category}")
        
        train_loss, val_loss, train_acc, val_acc = validate(
            model_name, emotion, category, k, num_people
        )
        
        history = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        
        # رسم نمودارها
        plot_training_history(history)
        
        # نمایش نتایج نهایی
        print(f"\n=== Final Results (averaged over {k} folds) ===")
        print(f"Average Train Loss: {np.mean(train_loss[-5:]):.4f}")
        print(f"Average Val Loss: {np.mean(val_loss[-5:]):.4f}")
        print(f"Average Train Accuracy: {np.mean(train_acc[-5:]):.2f}%")
        print(f"Average Val Accuracy: {np.mean(val_acc[-5:]):.2f}%")
        
        return history
        
    elif validation_type == 'subject_dependent':
        # حالت Subject-Dependent (وابسته به فرد)
        print(f"Running Subject-Dependent validation with {k}-fold cross-validation per subject...")
        print(f"Model: {model_name}, Emotion: {emotion}, Category: {category}")
        
        accuracies = choose_model(
            model_name, emotion, category, None, None,
            subject_dependecy='subject_dependent'
        )
        
        # محاسبه میانگین و واریانس
        avg_test_acc = np.sum(accuracies['test']) / num_people
        avg_train_acc = np.sum(accuracies['train']) / num_people
        
        _test_accs = np.array(accuracies['test'], dtype=float)
        _train_accs = np.array(accuracies['train'], dtype=float)
        var_test_acc = np.var(_test_accs, ddof=1)
        var_train_acc = np.var(_train_accs, ddof=1)
        
        # نمایش نتایج
        print(f"\n=== Final Results (averaged over {num_people} subjects) ===")
        print(f"Average Test Accuracy: {avg_test_acc:.2f}%")
        print(f"Average Train Accuracy: {avg_train_acc:.2f}%")
        print(f"Variance Test Accuracy: {var_test_acc:.6f}")
        print(f"Variance Train Accuracy: {var_train_acc:.6f}")
        
        # رسم نمودار
        plot_subject_dependet(accuracies)
        
        return {
            'test': accuracies['test'],
            'train': accuracies['train'],
            'avg_test_acc': avg_test_acc,
            'avg_train_acc': avg_train_acc,
            'var_test_acc': var_test_acc,
            'var_train_acc': var_train_acc
        }
    
    else:
        raise ValueError(
            f"Invalid validation_type: {validation_type}. "
            "Must be 'subject_independent' or 'subject_dependent'"
        )


def main():
    """
    تابع اصلی برای اجرا از command line
    """
    if len(sys.argv) < 6:
        print("Usage: python run.py <model_name> <emotion> <category> <k> <validation_type>")
        print("\nParameters:")
        print("  model_name      : simpleNN, cnn_45138, capsnet2020, hippoLegS1")
        print("  emotion         : valence, dominance")
        print("  category        : binary, 5category")
        print("  k               : number of folds (integer)")
        print("  validation_type : subject_independent, subject_dependent")
        print("\nExample:")
        print("  python run.py simpleNN valence binary 5 subject_independent")
        print("  python run.py simpleNN valence binary 5 subject_dependent")
        sys.exit(1)
    
    model_name = sys.argv[1]
    emotion = sys.argv[2]
    category = sys.argv[3]
    k = int(sys.argv[4])
    validation_type = sys.argv[5]
    
    # اجرای آزمایش
    results = run_experiment(model_name, emotion, category, k, validation_type)
    
    return results


if __name__ == "__main__":
    main()

