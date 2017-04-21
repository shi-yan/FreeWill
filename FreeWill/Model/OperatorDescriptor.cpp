#include "OperatorDescriptor.h"


FreeWill::OperatorDescriptor::OperatorDescriptor(const std::string &name,
        FreeWill::OperatorName operatorName,
        const std::map<std::string, FreeWill::TensorDescriptorHandle> &inputs,
        const std::map<std::string, FreeWill::TensorDescriptorHandle> &outputs,
        const std::map<std::string, std::any> &parameters,
        DataType dataType)
    :m_name(name),
      m_dataType(dataType),
      m_operatorName(operatorName),
      m_inputs(inputs),
      m_outputs(outputs),
      m_parameters(parameters)

{
}

FreeWill::OperatorDescriptor::~OperatorDescriptor()
{

}

void FreeWill::OperatorDescriptor::evaluateSVGDiagramSize(unsigned int &width, unsigned int &height)
{

    unsigned int maxAnchorItemCount = std::max(m_inputs.size(), m_outputs.size());

    height = maxAnchorItemCount * anchorHeight + (maxAnchorItemCount - 1) * anchorSpace + 2 * topBottomMargin + topSpace + bottomSpace;

    width = leftRightMargin * 2 + anchorWidth * 2 + centerSpace + leftSpace + rightSpace;

}

void FreeWill::OperatorDescriptor::generateSVGDiagram(std::ostream &outputStream, unsigned int &width, unsigned int &height)
{

    unsigned int maxAnchorItemCount = std::max(m_inputs.size(), m_outputs.size());

    height = maxAnchorItemCount * anchorHeight + (maxAnchorItemCount - 1) * anchorSpace + 2 * topBottomMargin + topSpace + bottomSpace;

    width = leftRightMargin * 2 + anchorWidth * 2 + centerSpace + leftSpace + rightSpace;

    outputStream << "<rect x=\"" << leftSpace
                 << "\" y=\"" << topSpace
                 << "\" width=\"" << leftRightMargin * 2 + anchorWidth * 2 + centerSpace
                 << "\" height=\"" << maxAnchorItemCount * anchorHeight + (maxAnchorItemCount - 1) * anchorSpace + 2 * topBottomMargin
                 << "\" rx=\"15\" ry=\"15\" style=\"fill:red;stroke:black;stroke-width:1;opacity:0.5\" />";

    outputStream << "<text x=\"" << width * 0.5 << "\" y=\"15\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"12\">";
    outputStream << m_name;
    outputStream << "</text>";

    unsigned int inputTopMargin = (height - topSpace - bottomSpace - (m_inputs.size() * anchorHeight + (m_inputs.size() - 1) * anchorSpace) ) * 0.5;
    unsigned int outputTopMargin = (height - topSpace - bottomSpace - (m_outputs.size() * anchorHeight + (m_outputs.size() - 1) * anchorSpace) ) * 0.5;

    unsigned int i = 0;
    for(auto inputIter = m_inputs.begin(); inputIter != m_inputs.end(); ++inputIter, ++i)
    {

        outputStream << "<rect x=\"" << leftSpace + leftRightMargin
                     << "\" y=\"" << inputTopMargin + topSpace + (anchorHeight + anchorSpace)*i
                     << "\" width=\"" << anchorWidth
                     << "\" height=\"" << anchorHeight
                     << "\" rx=\""<< anchorHeight * 0.5 <<"\" ry=\"" << anchorHeight*0.5 << "\" style=\"fill:blue;stroke:black;stroke-width:1;opacity:0.5\" />";

        outputStream << "<text x=\"" << leftSpace + leftRightMargin + anchorWidth * 0.5
                     << "\" y=\"" << inputTopMargin + topSpace + (anchorHeight + anchorSpace)*i + anchorHeight*0.5
                     <<"\" alignment-baseline=\"middle\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"12\">";
        outputStream << inputIter->first;
        outputStream << "</text>";
    }

    i = 0;
    for(auto outputIter = m_outputs.begin(); outputIter != m_outputs.end(); ++outputIter, ++i)
    {

        outputStream << "<rect x=\"" << width - rightSpace - leftRightMargin - anchorWidth
                     << "\" y=\"" << outputTopMargin + topSpace + (anchorHeight + anchorSpace)*i
                     << "\" width=\"" << anchorWidth
                     << "\" height=\"" << anchorHeight
                     << "\" rx=\""<< anchorHeight * 0.5 <<"\" ry=\"" << anchorHeight*0.5 << "\" style=\"fill:blue;stroke:black;stroke-width:1;opacity:0.5\" />";

        outputStream << "<text x=\"" << width - rightSpace - leftRightMargin - anchorWidth * 0.5
                     << "\" y=\"" << outputTopMargin + topSpace + (anchorHeight + anchorSpace)*i + anchorHeight*0.5
                     <<"\" alignment-baseline=\"middle\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"12\">";
        outputStream << outputIter->first;
        outputStream << "</text>";
    }

}
