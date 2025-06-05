from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
from utils.bert import calculate_semantic_similarity
from utils.icd10 import ICD10Taxonomy

"""
example of first 3 rows:
id,case,golden-diagnosis,golden-diagnosis-long,diagnostic_code/s,death,critical,pediatric,severity,icd10_diagnosis_name,icd10_chapter_code,icd10_block_code,icd10_category_code,icd10_category_name,icd10_disease_group_code,icd10_disease_group_name,icd10_disease_code,icd10_disease_name,icd10_disease_variant_code
0,"Motivo de consulta:\nPaciente acude a consulta para ser diagnosticado\n\nAnamnesis:\n\nPaciente Hombre de 17 años. Según refiere el paciente presenta en la noche del día de la fecha traumatismo a nivel de la mano derecha mientras practicaba actividad deportiva. Evoluciona con dolor y tumefacción locales por lo que acude.-\n\nAntecedentes:\n\nNo hay antecedentes\n\nExploracion:\n\nLúcido, reactivo, buen estado general. Adecuada coloración e hidratación de piel y mucosas. Deambula por sus medios sin alteración de la marcha, disnea ni taquipnea. Adecuada perfusión periférica. Eupneico. Habla fluida. Buena mecánica ventilatoria. Presenta tumefacción de partes blandas así como dolor a nivel del 2-4º MTC derechos. Motilidad de la muñeca conservada, indolora. No alteraciones neurovasculares distales.-\n\nPruebas clinicas:\n\n-Complementarias:\nRx mano derecha AP y oblic: Fractura diafisaria 3º MTC derecho.-\n",Fractura 3º metacarpo derecho,"Unspecified fracture of third metacarpal bone, right hand, initial encounter for closed fracture also known as Fractura 3º metacarpo derecho.",S62.302A,0,0,0,0,"Unspecified fracture of third metacarpal bone, right hand, initial encounter for closed fracture","XIX Injury, poisoning and certain other consequences of external causes",S60-S69 Injuries to the wrist and hand,S62,Fracture at wrist and hand level,S62.3,Fracture of other and unspecified metacarpal bone,S62.30,Unspecified fracture of other metacarpal bone,S62.302A
1,"Motivo de consulta:\nPaciente acude a consulta para ser diagnosticado\n\nAnamnesis:\n\nPaciente Hombre de 46 años. Según refiere el paciente presenta desde hace 15 días aproximadamente cuadro de otalgia derecha progresiva acompañado de pérdida de la calidad auditiva. No fiebre; discreta otorrea. En la fecha peoría del dolor, que no cede pese al tto con ibuprofeno, por lo que acude.-\n\nAntecedentes:\n\nNo hay antecedentes\n\nExploracion:\n\nLúcido, reactivo, buen estado general. Adecuada coloración e hidratación de piel y mucosas. Deambula por sus medios sin alteración de la marcha, disnea ni taquipnea. Adecuada perfusión periférica. Eupneico. Habla fluida. Buena mecánica ventilatoria. Otoscopia izquierda: s/p. Otoscopia derecha: CAE marcadamente hiperémico, no secretor. Membrana timpánica hiperémica y discretamente despulida. Trago positivo.-\n\nPruebas clinicas:\n\n-Complementarias:\nnan\n",Otitis externa derecha.\nOtitis media aguda derecha,"Unspecified otitis externa, right ear also known as Otitis externa derecha.\nOtitis media aguda derecha.",H60.91,0,0,0,0,"Unspecified otitis externa, right ear",VIII Diseases of the ear and mastoid process,H60-H62 Diseases of external ear,H60,Otitis externa,H60.9,Unspecified otitis externa,H60.91,"Unspecified otitis externa, right ear",
2,"Motivo de consulta:\nPaciente acude a consulta para ser diagnosticado\n\nAnamnesis:\n\nPaciente Hombre de 47 años. Según refiere el paciente presenta desde ayer cuadro de dolor anal en contexto de hemorroides. No rectorragia. No ha recibido tto analgésico. Ante la progresión del dolor acude para valoración.-\n\nAntecedentes:\n\nNo hay antecedentes\n\nExploracion:\n\nLúcido, reactivo, buen estado general. Adecuada coloración e hidratación de piel y mucosas. Deambula por sus medios sin alteración de la marcha, disnea ni taquipnea. Adecuada perfusión periférica. Eupneico. Habla fluida. Buena mecánica ventilatoria. Hemorroides externas congestivas sin signos de trombosis ni sangrado.-\n\nPruebas clinicas:\n\n-Rapidas:\n\nTensión arterial: 129.0 / 79.0\n\nFrecuencia cardiaca: 96.0\n\nTemperatura: 36.7\n\nSaturación de oxígeno: 96.0\n\n-Complementarias:\nnan\n",Congestión hemorroidal,Unspecified hemorrhoids also known as Congestión hemorroidal.,K64.9,0,0,0,0,Unspecified hemorrhoids,XI Diseases of the digestive system,K55-K64 Other diseases of intestines,K64,Hemorrhoids and perianal venous thrombosis,K64.9,Unspecified hemorrhoids,,,
"""

data_path = Path("before.csv")

# do not change the code above
# code below, properly implement the following:
# - we want to create a new csv file with the following columns:
#   - id
#   - case
#   - diagnosis (golden-diagnosis already)
#   - icd10_diagnosis (icd10_diagnosis_name)
#   - category: <icd10_category_name>
#   - block: <icd10_disease_group_name>
#   - sub_block: <icd10_disease_name>
#   - diagnostic_code (from all the columns which names finish as "_code" currently take the value of the one with a longer value)
#   - death
#   - critical
#   - pediatric
#   - severity

def find_longest_diagnostic_code(row: pd.Series) -> str:
    """Find the diagnostic code with the longest value from all columns ending with '_code'"""
    code_columns = [col for col in row.index if col.endswith('_code')]
    longest_code = ""
    
    for col in code_columns:
        code_value = str(row[col]) if pd.notna(row[col]) else ""
        if len(code_value) > len(longest_code):
            longest_code = code_value
    
    return longest_code

def transform_dataset(input_path: Path, output_path: Path) -> None:
    """Transform the dataset to the required format"""
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Processing {len(df)} rows...")
    
    # Create the new dataset with required columns
    new_df = pd.DataFrame({
        'id': df['id'],
        'case': df['case'],
        'diagnosis': df['golden-diagnosis'],
        'icd10_diagnosis': df['icd10_diagnosis_name'],
        'category': df['icd10_category_name'],
        'block': df['icd10_disease_group_name'],
        'sub_block': df['icd10_disease_name'],
        'diagnostic_code': df.apply(find_longest_diagnostic_code, axis=1),
        'death': df['death'],
        'critical': df['critical'],
        'pediatric': df['pediatric'],
        'severity': df['severity']
    })
    
    print(f"Saving transformed data to {output_path}...")
    new_df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(new_df)} rows to {output_path}")
    
    # Display summary statistics
    print("\nDataset Summary:")
    print(f"Total cases: {len(new_df)}")
    print(f"Deaths: {new_df['death'].sum()}")
    print(f"Critical cases: {new_df['critical'].sum()}")
    print(f"Pediatric cases: {new_df['pediatric'].sum()}")
    print(f"Average severity: {new_df['severity'].mean():.2f}")
    
    # Show unique categories
    print(f"\nUnique categories: {new_df['category'].nunique()}")
    print(f"Unique blocks: {new_df['block'].nunique()}")
    print(f"Unique sub-blocks: {new_df['sub_block'].nunique()}")

def main():
    """Main execution function"""
    input_file = data_path
    output_file = Path("transformed_dataset.csv")
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found!")
        return
    
    try:
        transform_dataset(input_file, output_file)
        print(f"\nTransformation completed successfully!")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
    except Exception as e:
        print(f"Error during transformation: {e}")
        raise

if __name__ == "__main__":
    main()
