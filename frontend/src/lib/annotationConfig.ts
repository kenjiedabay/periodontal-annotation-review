import type { AnnotationSeverity, DiseaseStatus } from '../types';

export const annotationLabels: {
  diseaseStatuses: readonly DiseaseStatus[];
  severities: readonly AnnotationSeverity[];
  findings: readonly string[];
  fdiTeeth: readonly string[];
} = {
  diseaseStatuses: ['Present', 'Absent', 'Uncertain'],
  severities: ['Mild', 'Moderate', 'Severe', 'Cannot determine'],
  findings: ['Alveolar bone loss', 'Crestal bone changes', 'Periodontal ligament space changes', 'Other expert-defined finding'],
  fdiTeeth: ['11', '12', '13', '14', '15', '16', '17', '18', '21', '22', '23', '24', '25', '26', '27', '28', '31', '32', '33', '34', '35', '36', '37', '38', '41', '42', '43', '44', '45', '46', '47', '48'],
};
