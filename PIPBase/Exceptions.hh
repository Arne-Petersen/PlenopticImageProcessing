/**
 * Copyright 2019 Arne Petersen, Kiel University
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
 *    associated documentation files (the "Software"), to deal in the Software without restriction, including
 *    without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
 *    sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject
 *    to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in all copies or substantial
 *    portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
 *    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
 *    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 *    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cstring>
#include <string>
#include <exception>

namespace PIP
{

///
/// \brief The CBasicException wraps std::exception for local instantiation. (std::exception has no CTor)
///
class CBasicException : public std::exception
{
public:
    ///
    /// \brief CBasicException creates excpetion with empty message
    ///
    CBasicException() : m_strErrorMessage("") {}

    ///
    /// \brief CBasicException creates an exception with given text message
    /// \param strMessage message to throw
    ///
    CBasicException(const std::string& strMessage)
        : m_strErrorMessage(strMessage)
    {}

    ///
    /// \brief CBasicException wraps the provided CBasicException with additional error message
    /// \param exc exception to wrap
    /// \param strMessage additional message to throw
    ///
    CBasicException(const CBasicException& exc, const std::string& strMessage = "")
        : std::exception(exc),
          m_strErrorMessage(strMessage + std::string("\nInner exception: ") + std::string(exc.what()))
    {}

    ///
    /// \brief CBasicException wraps the provided std::exception with addition error message
    /// \param exc exception to wrap
    /// \param strMessage additional message to throw
    ///
    CBasicException(const std::exception& exc, const std::string& strMessage = "")
        : std::exception(exc),
          m_strErrorMessage(strMessage + std::string("\nSystem exception : ") + std::string(exc.what()))
    {}

    /// Empty DTor
    virtual ~CBasicException() {}

    ///
    /// \brief what implements std::exception interface to \ref GetMessage
    /// \return exception message
    ///
    virtual const char* what() const throw() {return m_strErrorMessage.c_str();}

    ///
    /// \brief GetFullMessage returns accumulated (if re-thrown) exception message.
    /// \return exception message
    ///
    const std::string& GetFullMessage() const
    { return m_strErrorMessage; }

protected:

    /// Exception message
    std::string m_strErrorMessage;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
///          SPECIALIZED EXCEPTIONS
/////////////////////////////////////////////////////////////////////////////////////////////////////////////


///
/// \brief The ERuntimeExcpetionType enum represents the type of error, that caused an exception to be thrown.
///
enum class ERuntimeExcpetionType
{
    /// Exceptions source not known
    UNKNOWN = 593521,
    /// Tried to access protected or unavailable data
    ACCESS_VIOLATION,
    /// Tried to access a block interface. E.g. a thread is shutting down and rejects commands.
    INTERFACE_BLOCKED,
    ///
    MUTUALEXCLUSION,
    /// Type error during runtime (e.g. tried to debayer RGB image)
    ILLEGAL_ARGUMENT
};

///
/// \brief The CRuntimeException class provides exception with enum descriptor of occurred error
///        for advanced handling.
///
class CRuntimeException : public CBasicException
{
public:
    ///
    /// \brief CRuntimeException
    ///
    /// \param sMessage error message
    /// \param eExceptionType enum description of exception reason
    ///
    CRuntimeException(const std::string& sMessage = "",
                        ERuntimeExcpetionType eExceptionType=ERuntimeExcpetionType::UNKNOWN)
        : CBasicException(sMessage), m_eExceptionType(eExceptionType)
    {}

    ///
    /// \brief CRuntimeException wraps given std::exception with additional runtime information.
    ///
    /// \param exc exception to wrap
    /// \param sMessage error message
    /// \param eExceptionType enum description of exception reason
    ///
    CRuntimeException(const CBasicException& exc, const std::string& sMessage = "",
                        ERuntimeExcpetionType eExceptionType=ERuntimeExcpetionType::UNKNOWN)
        : CBasicException(std::string(exc.GetFullMessage()) + sMessage), m_eExceptionType(eExceptionType)
    {}

    ///
    /// \brief CRuntimeException wraps given std::exception with additional runtime information.
    ///
    /// \param exc exception to wrap
    /// \param sMessage error message
    /// \param eExceptionType enum description of exception reason
    ///
    CRuntimeException(const std::exception& exc, const std::string& sMessage = "",
                        ERuntimeExcpetionType eExceptionType=ERuntimeExcpetionType::UNKNOWN)
        : CBasicException(m_strErrorMessage + std::string("\n") + sMessage + std::string("\n -std ::") + std::string(exc.what())),
          m_eExceptionType(eExceptionType)
    {}

    virtual ~CRuntimeException() {}

    ///
    /// \brief what implements std::exception interface to 'GetMessage'
    /// \return exception message
    ///
    virtual const char* what() const throw()
    {
        return GetFullMessage().c_str();
    }

    ///
    /// \brief Type gets the error type descriptor.
    ///
    /// \return Enum describing exception reason
    ///
    inline ERuntimeExcpetionType Type() { return m_eExceptionType; }

protected:
    // Enum describing exception reason
    const ERuntimeExcpetionType m_eExceptionType;
};
}
